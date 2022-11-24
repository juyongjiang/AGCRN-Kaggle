import torch
import torch.nn as nn
import math
import os
import time
import copy
import numpy as np
import pandas as pd
from utils.util import get_logger
from utils.metrics import All_Metrics

class Trainer(object):
    def __init__(self, 
                 args,
                 final_result,
                 generator,
                 train_loader, val_loader, test_loader, predinput_data, scaler,
                 loss,
                 optimizer,
                 lr_scheduler):

        super(Trainer, self).__init__()
        self.args = args
        self.final_result = final_result
        self.num_nodes = args.num_nodes

        self.generator = generator
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler


        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.predinput_data = predinput_data
        self.scaler = scaler

        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)

        self.best_path = os.path.join(self.args.log_dir, f'best_model_{args.dataset}.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png') # when plot=True
        
        # log info
        if os.path.isdir(args.log_dir) == False:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info(f"Argument: {args}")
        for arg, value in sorted(vars(args).items()):
            self.logger.info(f"{arg}: {value}")

    def train_epoch(self, epoch):
        self.generator.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            batch_size = data.shape[0]
            data = data[..., :self.args.input_dim] # [B'', W, N, 1]
            label = target[..., :self.args.output_dim]  # # [B'', H, N, 1]

            #-------------------------------------------------------------------
            # Train Generator 
            #-------------------------------------------------------------------
            self.optimizer.zero_grad()
                        
            # data and target shape: B, W, N, F, and B, H, N, F; output shape: B, H, N, F (F=1)
            output = self.generator(data)
            if self.args.real_value: # it is depended on the output of model. If output is real data, the label should be reversed to real data
                label = self.scaler.inverse_transform(label)
                        
            loss = self.loss(output.cuda(), label)
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            # log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Generator Loss: {:.6f}'.format(
                                 epoch, 
                                 batch_idx, self.train_per_epoch, 
                                 loss.item()))
        
        train_epoch_loss = total_loss / self.train_per_epoch # average generator loss

        self.logger.info('**********Train Epoch {}: Averaged Generator Loss: {:.6f}'.format(epoch, train_epoch_loss))

        # learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        
        return train_epoch_loss

    def val_epoch(self, epoch, val_dataloader):
        self.generator.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data[..., :self.args.input_dim] # [B'', W, N, 1]
                label = target[..., :self.args.output_dim] # [B'', H, N, 1]
                output = self.generator(data)
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
                loss = self.loss(output.cuda(), label)
                # a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        
        return val_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []

        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            train_epoch_loss = self.train_epoch(epoch)
            
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
                
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs! Training stops!".format(self.args.early_stop_patience))
                    # break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.generator.state_dict())

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        # test: load the best model in the validation dataset
        self.generator.load_state_dict(best_model)
        self.save_checkpoint() # save the best model
        # self.val_epoch(self.args.epochs, self.test_loader)
        self.test(self.generator, self.args, self.predinput_data, self.scaler, self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.generator.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    # @staticmethod
    def test(self, model, args, predinput_data, scaler, logger, path=None):
        # calculate the number of auto-regressive prediction
        test_data = pd.read_csv('./dataset/test.csv')
        store_data = test_data[test_data['Store']==eval(args.dataset)]
        dept_data = store_data[['Dept', 'Date']].groupby('Dept')
        len_list = []
        for idx, (dept_id, sale_data) in enumerate(dept_data):
            len_list.append(len(sale_data))
        pred_len = max(len_list)
        rept_time = pred_len // args.horizon + 1

        cuda = True if torch.cuda.is_available() else False
        TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        predinput_data = TensorFloat(predinput_data).unsqueeze(0) # [1, T, N, 1]

        if path != None:
            check_point = torch.load(os.path.join(path, 'best_model.pth')) # path = args.log_dir
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        
        y_pred = []
        with torch.no_grad():
            for i in range(rept_time):
                output = model(predinput_data[:, -args.lag:, :, :]) # [1, T, N, 1] -> [1, H, N, 1]
                y_pred.append(output)
                predinput_data = torch.cat([predinput_data, output], dim=1) # [1, T+H, N, 1]
 
        if args.real_value:
            y_pred = torch.cat(y_pred, dim=1).squeeze() # [args.horizon*rept_time, N]
        else:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=1)).squeeze()
        
        # save predicted results as numpy format
        store_pred_data = y_pred.cpu().numpy().T # [N, args.horizon*rept_time]
        # print(len(len_list), store_pred_data.shape, pred_len)
        np.save(os.path.join(args.log_dir, 'store_{}_pred.npy'.format(args.dataset)), store_pred_data) 
        for dept_id, cut_len in enumerate(len_list):
            self.final_result.extend(list(store_pred_data[dept_id, :cut_len]))
