import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

class TimeSeriesDataset(Dataset):
    def __init__(self, 
                 currency_list,
                 x: np.ndarray, 
                 y: np.ndarray,
                 data_use_type,
                 train_percentage,
                 val_percentage,
                 test_percentage,
                 seq_len, 
                 ):
        self.currencies = currency_list
        self.n_currencies = len(self.currencies)
        self.x = torch.tensor(x[:self.n_currencies]).float()
        self.y = torch.tensor(y[:self.n_currencies]).long()
        self.seq_len = seq_len
        self.data_use_type = data_use_type
        
        
        #self.train_size = int(len(self.x[0]) * train_percentage)
        self.val_size = int(len(self.x[0]) * val_percentage)
        self.test_size = int(len(self.x[0]) * test_percentage)
        self.train_size = len(self.x[0]) - self.val_size - self.test_size 
        
        self.train_mean = [self.x[i][:self.train_size].mean() for i in range(self.n_currencies)]
        self.train_std = [self.x[i][:self.train_size].std() for i in range(self.n_currencies)]
        
#         self.train_min = [self.x[i][:self.train_size].min() for i in range(n_currencies)]
#         self.train_max = [self.x[i][:self.train_size].max() for i in range(n_currencies)]
        
    def __len__(self):
        
        if self.data_use_type == "train":
            return self.train_size - ( self.seq_len)

        elif self.data_use_type == "val":
            return self.val_size
  
        else:
            return self.test_size
        
    
    def __getitem__(self, index):
        
        item = dict()
        
        if self.data_use_type =="val":
            index = self.train_size + index - self.seq_len
            
        elif self.data_use_type =="test":
            index = self.train_size + self.val_size + index - self.seq_len
        
        for i in range(self.n_currencies):
            window = self.x[i][index:index+self.seq_len]
            window = (window -self.train_mean[i]) / self.train_std[i]
            
            item[self.currencies[i] + "_window"] = window
            item[self.currencies[i] + "_label"]  = self.y[i][index+self.seq_len]

        return item

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)
        
    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]
    
    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor