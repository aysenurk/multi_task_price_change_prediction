import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def get_data(currency_lst,
             n_classes,
             frequency, 
             window_size,
             neutral_quantile = 0.33,
             beg_date = pd.Timestamp(2013,1,1),
             end_date = pd.Timestamp.now(),
             log_price = True,
             remove_trend = True,
             include_indicators = False,
             include_imfs = False,
             open_high_low_volume = False):
        
        X, y, dfs = {}, {}, {}     
        
        for cur in currency_lst:
            df = pd.read_csv(f"../data/0_raw/Binance/{str.lower(cur)}_usdt_1d.csv", index_col=0)
            df.index = pd.to_datetime(df.index, unit='s')
            df.sort_index(inplace=True)
            #df.index = df.Date.apply(pd.Timestamp)
            #df.sort_values("Date", inplace=True)
            #df.set_index("Date", inplace=True)
            df.drop(["Date"], axis=1, inplace=True)
            df.rename(str.lower, axis=1, inplace=True)
            
#             if log_price:
#                 df[["close", "open", "high", "low"]] = df[["close", "open", "high", "low"]].apply(np.log, axis=1)
                   
#             if n_classes == 3:
#                 df['pct_diff'] = df['close'].pct_change()
#                 neutral_quantiles = df['pct_diff'].abs().quantile(neutral_quantile)
                
#                 conditions = [(df['pct_diff'] < 0) & (df['pct_diff'].abs() > neutral_quantiles),
#                               (df['pct_diff'] > 0) & (df['pct_diff'].abs() > neutral_quantiles)]

#                 classes = [0,1] # 2 is the default class if none of conditions is met, i.e. price change in the neutral range
            
#                 change_dir = np.select(conditions, classes, default=2)
            
#             else:
#                 df['diff'] = df['close'].diff()
#                 change_dir = df['diff'].apply(lambda x: 0 if x <= 0 else 1)
            
#             df.insert(loc=0, column="change_dir", value=change_dir)   
#             df.dropna(inplace=True)       
            
            if include_indicators:
                from ta import add_all_ta_features
                indicators_df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
                df[indicators_df.columns] = indicators_df
            #else:
            #    df.drop(["volume", "open", "high", "low"], axis=1, inplace=True)
            
            if include_imfs:
                from PyEMD import EEMD
                eemd = EEMD()
                imfs = eemd(df["close"].values)
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
                components = seasonal_decompose(df["close"], model="additive")
                df["close"] -= components.trend
                df.dropna(inplace=True)
                
            if not open_high_low_volume:
                df.drop(["open", "high", "low", "volume"], axis=1, inplace=True)

            dfs[cur] = df
        
        min_dates = [df.index.min() for cur, df in dfs.items()]
        max_dates = [df.index.max() for cur, df in dfs.items()]
        beg_date = max([max(min_dates), beg_date])
        end_date = min([min(max_dates), end_date])
        common_range = pd.date_range(beg_date, end_date, freq=frequency)
        
        diff_col = 'pct_diff' if n_classes == 3 else 'diff'
        X = np.array([dfs[cur].loc[common_range].drop(["change_dir", diff_col], axis=1).values for cur in currency_lst])
        y = np.array([dfs[cur].loc[common_range, "change_dir"].values for cur in currency_lst])
        features = df.columns.tolist()
        
        return X, y, features, dfs