
import os
import sys
import torch
import numpy as np
import torch.nn as nn
import argparse
import pandas as pd
import configparser
from datetime import datetime

from model.agcrn import AGCRN as Generator
from trainer import Trainer
from tqdm import tqdm
from dataloader import get_dataloader, get_predinput
from utils.metrics import MAE_torch
from utils.util import *

#*************************************************************************#
Mode = 'Train'
DEBUG = 'True'
MODEL = 'AGCRN-Kaggle'
#*************************************************************************#

# get configuration
config_file = './config/walmart.conf'
print('Reading configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)

def get_arguments():
    # parser
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--mode', default=Mode, type=str)
    parser.add_argument('--debug', default=DEBUG, type=eval)
    parser.add_argument('--model', default=MODEL, type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    # data
    parser.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
    parser.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
    parser.add_argument('--lag', default=config['data']['lag'], type=int)
    parser.add_argument('--horizon', default=config['data']['horizon'], type=int)
    parser.add_argument('--tod', default=config['data']['tod'], type=eval)
    parser.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
    parser.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
    parser.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
    # model
    parser.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    parser.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    parser.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    parser.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
    parser.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
    parser.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
    # train
    parser.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
    parser.add_argument('--seed', default=config['train']['seed'], type=int)
    parser.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    parser.add_argument('--epochs', default=config['train']['epochs'], type=int)
    parser.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    parser.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    parser.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    parser.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    parser.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
    parser.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    parser.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    parser.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    parser.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
    # test
    parser.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
    parser.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
    # log
    parser.add_argument('--log_dir', default='./', type=str)
    parser.add_argument('--log_step', default=config['log']['log_step'], type=int)
    parser.add_argument('--plot', default=config['log']['plot'], type=eval)
    args = parser.parse_args()
    for arg, value in sorted(vars(args).items()):
        print(f"{arg}: {value}")
    
    return args

def init_model(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    print_model_parameters(model, only_num=False)
    
    return model

if __name__ == "__main__":
    args = get_arguments()
    args.device = get_device(args)
    node_num = [74, 75, 67, 75, 70, 74, 72, 73, 69, 75, 73, 71, 75, 74, 73, 73, 73, 75, 74, 76, 70, 70, 73, 75, 
                73, 72, 74, 73, 71, 60, 75, 73, 56, 75, 70, 58, 62, 62, 72, 74, 73, 60, 57, 59, 71] # 45 stores
    final_result = []
    #=========================================================
    for i in tqdm(range(45)):
        print(f"===================================== Store {i+1} ===========================================")
        args.dataset = str(i+1)
        args.num_nodes = node_num[i]

        # init generator model
        generator = Generator(args)
        generator = generator.to(args.device)
        generator = init_model(generator)

        # load dataset X = [B', W, N, D], Y = [B', H, N, D]
        train_loader, val_loader, test_loader, scaler = get_dataloader(args,
                                                                    normalizer=args.normalizer,
                                                                    tod=args.tod, 
                                                                    dow=False,
                                                                    weather=False, 
                                                                    single=False)
        predinput_data = scaler.transform(get_predinput(args)) # [lag, N, 1]
        # loss function
        if args.loss_func == 'mask_mae':
            loss = masked_mae_loss(scaler, mask_value=0.0)
        elif args.loss_func == 'mae':
            loss = torch.nn.L1Loss().to(args.device)
        elif args.loss_func == 'mse':
            loss = torch.nn.MSELoss().to(args.device)
        else:
            raise ValueError

        # optimizer
        optimizer = torch.optim.Adam(params=generator.parameters(), 
                                    lr=args.lr_init, 
                                    eps=1.0e-8, 
                                    weight_decay=0, 
                                    amsgrad=False)
                                        
        # learning rate decay scheduler
        if args.lr_decay:
            print('Applying learning rate decay.')
            lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                                milestones=lr_decay_steps,
                                                                gamma=args.lr_decay_rate)
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)

        # config log path
        current_time = datetime.now().strftime('%Y%m%d%H%M%S')
        current_dir = os.path.dirname(os.path.realpath(__file__))
        log_dir = os.path.join(current_dir, 'log', args.dataset, current_time)
        args.log_dir = log_dir

        # model training or testing
        trainer = Trainer(args,
                        final_result,
                        generator, 
                        train_loader, val_loader, test_loader, predinput_data, scaler, 
                        loss, 
                        optimizer, 
                        lr_scheduler)
        
        if args.mode.lower() == 'train':
            trainer.train()
        elif args.mode.lower() == 'test':
            generator.load_state_dict(torch.load('./pre-trained/{}.pth'.format(args.dataset)))
            print("Load saved model")
            trainer.test(generator, trainer.args, predinput_data, scaler, trainer.logger)
        else:
            raise ValueError
    #=========================================================
    # generate the final submission files
    test_data = pd.read_csv('./dataset/test.csv')
    submission = pd.DataFrame({
        "Id": test_data.Store.astype(str)+'_'+test_data.Dept.astype(str)+'_'+test_data.Date.astype(str),
        "Weekly_Sales": final_result
    })

    submission.to_csv('submission_agcrn_kaggle.csv', index=False)