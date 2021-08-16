import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from datetime import datetime

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
        
        self.train_size = int(len(self.x[0]) * train_percentage)
        self.val_size = int(len(self.x[0]) * val_percentage)
        self.test_size = len(self.x[0]) - self.train_size - self.val_size
        
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

def get_data(currency_lst,
             data_frequency,
             pred_frequency, 
             n_classes,
             window_size,
             neutral_quantile = 0.33,
             beg_date = pd.Timestamp(2013,1,1),
             end_date = pd.Timestamp.now(),
             log_price = True,
             remove_trend = False,
             ma_period = 0,
             include_indicators = False,
             include_imfs = False,
             open_high_low_volume = False,
             drop_missing = True):
 
        X, y, dfs = {}, {}, {}     
        
        for cur in currency_lst:
            df = pd.read_csv(f"../data/0_raw/Binance/{str.lower(cur)}_usdt_{data_frequency}.csv", header=None,index_col=0)
            try: #for the previous raw data format in the project
                df.index = pd.to_datetime(df.index, unit='s')
                df.drop(["Date"], axis=1, inplace=True)
                df.rename(str.lower, axis=1, inplace=True) 
            except: #for the current raw data format in the project
                df.index = pd.to_datetime(df.index/1000, unit='s')
                df.sort_index(inplace=True)
                df.columns = ["open","high","low","close","volume"]
            
            if include_indicators:
                from ta import add_all_ta_features
                indicators_df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
                df[indicators_df.columns] = indicators_df
            
            if include_imfs:
                from PyEMD import EEMD
                eemd = EEMD(parallel=True, processes=2)
                imfs = eemd(df["close"].values, max_imf=7)
                imf_features = ["imf_"+str(i) for i in range(imfs.shape[0])]
                df = pd.concat((df, pd.DataFrame(imfs.T, columns=imf_features, index=df.index)), axis=1)
            
            if log_price:
                df[["close", "open", "high", "low"]] = df[["close", "open", "high", "low"]].apply(np.log, axis=1)
                   
            if n_classes == 3:
                df['pct_diff'] = df['close'].pct_change()
                neutral_quantiles = df['pct_diff'].abs().quantile(neutral_quantile)
                
                conditions = [(df['pct_diff'] < 0) & (df['pct_diff'].abs() > neutral_quantiles),
                              (df['pct_diff'] > 0) & (df['pct_diff'].abs() > neutral_quantiles)]

                classes = [0,1] # 2 is the default class if none of conditions is met, i.e. price change in the neutral range
            
                change_dir = np.select(conditions, classes, default=2)
            
            else:
                df['diff'] = df['close'].diff()
                change_dir = df['diff'].apply(lambda x: 0 if x <= 0 else 1)
            
            df.insert(loc=0, column="change_dir", value=change_dir)   
            df.dropna(inplace=True)  
            
            if remove_trend:
                from statsmodels.tsa.seasonal import seasonal_decompose
                components = seasonal_decompose(df["close"], model="additive", period = ma_period, two_sided=False)
                df["close"] -= components.trend
                df.dropna(inplace=True)
                
            if not open_high_low_volume:
                df.drop(["open", "high", "low", "volume"], axis=1, inplace=True)

            dfs[cur] = df
        
        min_dates = [df.index.min() for cur, df in dfs.items()]
        max_dates = [df.index.max() for cur, df in dfs.items()]
        beg_date = max([max(min_dates), beg_date])
        end_date = min([min(max_dates), end_date])
        common_range = pd.date_range(beg_date, end_date, freq=pred_frequency)
        
        if drop_missing:
            #the following time steps are missing in 6h datasets
            missing = ['2018-02-08 06:00:00', '2018-02-08 12:00:00', '2018-02-08 18:00:00', '2018-02-09 00:00:00', '2018-06-26 06:00:00', '2019-05-15 06:00:00']
            str_to_timestamp = lambda str_timestamp: datetime.strptime(str_timestamp, "%Y-%m-%d %H:%M:%S") 
            missing = list(map(str_to_timestamp, missing))
            common_range = common_range.drop(missing)
        
        diff_col = 'pct_diff' if n_classes == 3 else 'diff'

        X = np.array([dfs[cur].loc[common_range].drop(["change_dir", diff_col], axis=1).values for cur in currency_lst])
        y = np.array([dfs[cur].loc[common_range, "change_dir"].values for cur in currency_lst])
        features = df.columns.tolist()
        
        return X, y, features, dfs