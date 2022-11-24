import os
import numpy as np
import torch
import torch.utils.data
from utils.norm import *


def get_dataloader(args, normalizer='std', tod=False, dow=False, weather=False, single=True):
    # load raw st dataset
    data = load_st_dataset(args.dataset)        # [B, N, 1] B means the number of entries
    
    # normalize st data
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
   
    # spilit dataset by ratio
    data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    
    # add time window [B, N, 1]
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    
    ##############get dataloader######################
    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    if len(x_test) == 0:
        test_dataloader = None
    else:
        test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler

def get_predinput(args):
    # load raw st dataset
    data = load_st_dataset(args.dataset)       # [B, N, 1] B means the number of entries
    predinput_data = data[-args.lag:]   # [lag, N, 1]

    return predinput_data

def load_st_dataset(store_id):
    data_path = f'./dataset/processed/store_{store_id}.npy'
    data = np.load(data_path).T # transpose -> [B, N]
    
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1) # [B, N, 1]
    print('Load %s Dataset shaped: ' % store_id, data.shape, data.max(), data.min(), data.mean(), np.median(data))

    return data


def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        # column min max, to be depressed
        # note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError

    return data, scaler


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0] # T
    test_data = data[-int(data_len*test_ratio):] # not necessary in this competition
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data # some departments only have a few historical sales data (e.g., <10), thus we consider all data in the training stage

    return train_data, val_data, test_data


def Add_Window_Horizon(data, window=12, horizon=3, single=False):
    '''
    Historical 12 steps to forecaste the next 3 steps
    :param data: shape [B, N, D]
    :param window:
    :param horizon:
    :return: X is [B', W, N, D], Y is [B', H, N, D], B' = B - W - H + 2
    '''
    length = len(data) 
    end_index = length - horizon - window + 1
    X = []      # windows
    Y = []      # horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)

    return X, Y


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)

    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, 
                                             batch_size=batch_size,
                                             shuffle=shuffle, 
                                             drop_last=drop_last)
    return dataloader
