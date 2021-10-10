import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import  Dataset

dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", ".."
    )
)
#beginning and ending dates of common range of BTC, ETH, LTC, ADA, and XRP for both 6h and 1h datasets
FIRST_DATE = '2018-05-04'
LAST_DATE = '2021-08-13'

class TimeSeriesDataset(Dataset):
    def __init__(self, 
                 x: np.ndarray, 
                 y: np.ndarray,
                 data_use_type,
                 currency_list,
                 dataset_percentages,
                 window_size, #in terms of days
                 data_frequency,
                 pred_frequency,
                 **kwargs
                 ):
        self.currencies = currency_list
        self.n_currencies = len(currency_list)
        self.data_use_type = data_use_type

        if data_frequency == pred_frequency:
            self.freq_multiplier = 1

        elif data_frequency == '1h' and pred_frequency == '1d':
            self.freq_multiplier = 24
                  
        elif data_frequency == '6h' and pred_frequency == '1d':
            self.freq_multiplier = 4

        else:
            raise Exception(f"Improper frequency setting: {data_frequency}, {pred_frequency}")
            
        self.seq_len = window_size * self.freq_multiplier
        self.pred_indicies = np.arange(0, len(x[0]), self.freq_multiplier)
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).long()
        
        _, val_percentage, test_percentage = dataset_percentages
        self.total_size = len(self.pred_indicies)
        self.val_size = int(self.total_size * val_percentage)
        self.test_size = int(self.total_size * test_percentage)
        self.train_size = self.total_size - self.val_size - self.test_size 

        self.train_mean = [self.x[i][:self.train_size].mean(axis=0) for i in range(self.n_currencies)]
        self.train_std = [self.x[i][:self.train_size].std(axis=0) for i in range(self.n_currencies)]

 
    def __len__(self):
        
        if self.data_use_type == "train":
            return int(self.train_size - self.seq_len / self.freq_multiplier)

        elif self.data_use_type == "val":
            return self.val_size

        else:
            return self.test_size
    
    def __getitem__(self, index):
        
        item = dict()
        
        if self.data_use_type =="val":
            index = self.train_size + index * self.freq_multiplier - self.seq_len
            
        elif self.data_use_type =="test":
            index = self.train_size + self.val_size + index * self.freq_multiplier - self.seq_len
        else:
            index *= self.freq_multiplier
        
        for i in range(self.n_currencies):
            window = self.x[i][index:index+self.seq_len]
         
            if self.data_use_type != "train":
                mean = self.x[i][:index+self.seq_len].mean(axis=0)
                std = self.x[i][:index+self.seq_len].std(axis=0)
                window = (window - mean) / (std + 0.00001)
            else:
                window = (window - self.train_mean[i]) / (self.train_std[i] + 0.00001)
            
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

def get_data(currency_list,
             data_frequency,
             pred_frequency, 
             num_classes,
             neutral_quantile = 0.33,
             beg_date = FIRST_DATE,
             end_date = LAST_DATE,
             log_price = True,
             remove_trend = False,
             decompose = False,
             ma_period = 7, #in terms of days
             indicators = False,
             imfs = False,
             ohlv = False,
              **kwargs):

        beg_date = pd.Timestamp(beg_date)
        end_date = pd.Timestamp(end_date)

        X, y, dfs = {}, {}, {}     
    
        for cur in currency_list:
            df = pd.read_csv(dir+f"/data/0_raw/Binance/{str.lower(cur)}_usdt_{data_frequency}.csv", header=None,index_col=0)
            df.index = pd.to_datetime(df.index, unit='ms')
            df.sort_index(inplace=True)
            df.columns = ["open","high","low","close","volume"]
            
            if indicators:
                from ta import add_all_ta_features
                indicators_df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
                df[indicators_df.columns] = indicators_df
          
            if imfs:
                from PyEMD import EEMD
                eemd = EEMD(parallel=True, processes=2)
                imfs_result = eemd(df["close"].values, max_imf=7)
                imf_features = ["imf_"+str(i) for i in range(imfs_result.shape[0])]
                df = pd.concat((df, pd.DataFrame(imfs_result.T, columns=imf_features, index=df.index)), axis=1)
            
            if log_price:
                df[["close", "open", "high", "low"]] = df[["close", "open", "high", "low"]].apply(np.log, axis=1)
                   
            if num_classes == 3:
                pct_diff = df['close'].pct_change()
                quantile_value = pct_diff.abs().quantile(neutral_quantile).loc[neutral_quantile]
                
                conditions = [(pct_diff < 0) & (pct_diff.abs() > quantile_value),
                              (pct_diff > 0) & (pct_diff.abs() > quantile_value)]

                classes = [0,1] # 2 is the default class if none of conditions is met, i.e. price change in the neutral range
            
                change_dir = np.select(conditions, classes, default=2)
            
            else: 
                change_dir = df['close'].diff().apply(lambda x: 0 if x <= 0 else 1)
            
            df.insert(loc=0, column="change_dir", value=change_dir)   
            
            if remove_trend:
                #from statsmodels.tsa.seasonal import seasonal_decompose
                #ma_period = ma_period if pred_frequency in ['d', 'D'] else ma_period * 4
                #components = seasonal_decompose(df["close"], model="additive", period = ma_period, two_sided=False)
                #df["close"] -= components.trend
                df['diff'] = df['close'].diff()
                #df['diff'] = df['close'].pct_change()
                if not decompose:
                    df.drop('close', axis=1, inplace=True)  

            if decompose: 
                from statsmodels.tsa.seasonal import seasonal_decompose
                ma_period = ma_period if pred_frequency == '1d' else ma_period * 4 #if pred_frequency is 6h, then multiply the ma_period by 4 
                components = seasonal_decompose(df["close"], model="additive", period = ma_period, two_sided=False)
                df['trend'] = components.trend
                df['residual'] = components.resid  
                df['seasonal'] = components.seasonal

            if ohlv: #keeping open, high, low, and volume
                df[['open_d', 'high_d', 'low_d', 'volume_d']] = df[["open", "high", "low", "volume"]].diff() 
            else:
                df.drop(["open", "high", "low", "volume"], axis=1, inplace=True)
            
            dfs[cur] = df.dropna()
        
        min_dates = [df.index.min() for cur, df in dfs.items()]
        max_dates = [df.index.max() for cur, df in dfs.items()]
        beg_date = max([max(min_dates), beg_date])
        end_date = min([min(max_dates), end_date])
        common_range = pd.date_range(beg_date, end_date, freq=data_frequency)
        
        missing = set()
        common_set = set(common_range)
        for cur, df in dfs.items():
            missing_steps = common_set.difference(df.index)
            missing |= missing_steps
        common_range = common_range.drop(missing)
        
        X = np.array([dfs[cur].loc[common_range].drop(["change_dir"], axis=1).values for cur in currency_list])
        y = np.array([dfs[cur].loc[common_range, "change_dir"].values for cur in currency_list])
        features = df.columns.tolist()
        features.remove("change_dir")

        return X, y, features, dfs, common_range