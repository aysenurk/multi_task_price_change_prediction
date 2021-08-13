import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from TimeSeriesLearningUtils import get_data
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-cl", "--currency-list", nargs="+", default=[])
parser.add_argument("-lstm", "--lstm-list", nargs="+", default=[])
parser.add_argument("-trend", default=bool)
parser.add_argument("-imfs", default=bool)
parser.add_argument("-ohlv", default=bool)
parser.add_argument("-bidirectional", default=bool)
parser.add_argument("-indicators", default=bool)
parser.add_argument("-weight", default=bool)
parser.add_argument("-classes", default=int)

args = parser.parse_args()
n_classes = int(args.classes)
currency_list = args.currency_list
remove_trend = bool(int(args.trend))
imfs = bool(int(args.imfs))
indicators =  bool(int(args.indicators))
ohlv = bool(int(args.ohlv))
bidirectional = bool(int(args.bidirectional))
loss_weight_calculate = bool(int(args.weight))
lstm_hidden_sizes = [int(size) for size in args.lstm_list]

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
        
        self.train_mean = [self.x[i][:self.train_size].mean(axis=0) for i in range(self.n_currencies)]
        self.train_std = [self.x[i][:self.train_size].std(axis=0) for i in range(self.n_currencies)]
        
        
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
 
class LSTM_based_classification_model(pl.LightningModule):
    def __init__(self,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 calculate_loss_weights,
                 currencies,
                 num_classes,
                 window_size,
                 input_size,
                 batch_size,
                 lstm_hidden_sizes,
                 bidirectional,
                 learning_rate = 1e-3,
                 scheduler_step = 10,
                 scheduler_gamma = 0.1,
                 ):
        
        super().__init__()
        self.num_classes = num_classes
        self.currencies = currencies
        self.num_tasks = len(currencies)
        self.window_size = window_size
        self.input_size = input_size
        self.batch_size = batch_size
        
        self.lstm_hidden_sizes = lstm_hidden_sizes
        self.bidirectional = bidirectional 
        
        if calculate_loss_weights:
            loss_weights = []
            for i in range(self.num_tasks):
                train_labels = [int(train_dataset[n][self.currencies[i] +"_label"] )for n in range(train_dataset.__len__())]
                samples_size = pd.DataFrame({"label": train_labels}).groupby("label").size().to_numpy()
                loss_weights.append((1 / samples_size) * sum(samples_size)/2)
            self.weights = loss_weights
        else:
            self.weights = None
        
        self.lstm_1 = nn.LSTM(input_size = self.input_size, 
                              num_layers=1, 
                              batch_first=True, 
                              hidden_size = self.lstm_hidden_sizes[0], 
                              bidirectional = bidirectional)
        self.batch_norm1 = nn.BatchNorm2d(num_features=self.lstm_hidden_sizes[0] * 2 if bidirectional else self.lstm_hidden_sizes[0])
        
        if len(self.lstm_hidden_sizes) > 1:
            self.lstm_2 = nn.LSTM(input_size = self.lstm_hidden_sizes[0] *2 if self.bidirectional else self.lstm_hidden_sizes[0], 
                                  num_layers=1, 
                                  batch_first=True, 
                                  hidden_size = self.lstm_hidden_sizes[1] , 
                                  bidirectional = bidirectional)
            self.batch_norm2 = nn.BatchNorm2d(num_features=self.lstm_hidden_sizes[1] *2 if self.bidirectional else self.lstm_hidden_sizes[1])

            self.lstm_3 = nn.LSTM(input_size = self.lstm_hidden_sizes[1] *2 if self.bidirectional else self.lstm_hidden_sizes[1], 
                                  num_layers=1, 
                                  batch_first=True, 
                                  hidden_size = self.lstm_hidden_sizes[2] , 
                                  bidirectional = bidirectional)
            self.batch_norm3 = nn.BatchNorm2d(num_features=self.lstm_hidden_sizes[2] *2 if self.bidirectional else self.lstm_hidden_sizes[2])
        
        self.dropout = nn.Dropout(0.5)
        n_feature = self.lstm_hidden_sizes[-1] *2 if bidirectional else self.lstm_hidden_sizes[-1]
        
        self.linear1 =[nn.Linear(n_feature, int(n_feature/2))] * self.num_tasks
        self.linear1 = torch.nn.ModuleList(self.linear1)
        self.activation = nn.ReLU()
        
        self.output_layers = [nn.Linear(int(n_feature/2), self.num_classes)] * self.num_tasks
        self.output_layers = torch.nn.ModuleList(self.output_layers)
        
        if self.weights != None:
            self.cross_entropy_loss = [nn.CrossEntropyLoss(weight= torch.tensor(weights).float()) for weights in self.weights]
        else:
            self.cross_entropy_loss = [nn.CrossEntropyLoss() for _ in range(self.num_tasks)]
        
        self.cross_entropy_loss = torch.nn.ModuleList(self.cross_entropy_loss)
        
        self.f1_score = pl.metrics.F1(num_classes=self.num_classes, average="macro")
        self.accuracy_score = pl.metrics.Accuracy()
        
        self.train_dl = DataLoader(train_dataset, batch_size=self.batch_size, shuffle = True)
        self.val_dl = DataLoader(val_dataset, batch_size=self.batch_size)
        self.test_dl = DataLoader(test_dataset, batch_size=self.batch_size)
        
        self.learning_rate = learning_rate
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma
        
    def forward(self, x, i):

        batch_size = x.size()[0]
        
        x = x.view(batch_size, self.window_size, self.input_size) #(batch, window_len, feature_size)
        x, _  = self.lstm_1(x)
        
        x = self.dropout(x)

        x = x.reshape(x.size()[-1], batch_size, self.window_size) #(feature_size, batch, window_len)
        x = self.batch_norm1(x.unsqueeze(0))
        
        if len(self.lstm_hidden_sizes) > 1:
            
            x = x.view(batch_size, self.window_size, x.size()[1])
            x, _  = self.lstm_2(x)

            x = self.dropout(x)

            x = x.reshape(x.size()[-1], batch_size, self.window_size) #(feature_size, batch, window_len)
            x = self.batch_norm2(x.unsqueeze(0))

            x = x.view(batch_size, self.window_size, x.size()[1])
            x, _  = self.lstm_3(x)

            x = self.dropout(x)

            x = x.reshape(x.size()[-1], batch_size, self.window_size) #(feature_size, batch, window_len)
            x = self.batch_norm3(x.unsqueeze(0))
        
        x = x.view(batch_size, self.window_size, x.size()[1])
        x = x[:, -1, :] # equivalent to return sequence = False on keras :)
        
        x = self.dropout(x)
        
        x = self.linear1[i](x)
        x = self.activation(x)
                 
        output = self.output_layers[i](x)
    
        return output
    
    
    def training_step(self, batch, batch_nb):
        
        loss = (torch.tensor(0.0, device="cuda:0", requires_grad=True) + \
                torch.tensor(0.0, device="cuda:0", requires_grad=True)) 
        # araÅŸtÄ±rÄ±labilir
        for i in range(self.num_tasks):
            x, y = batch[self.currencies[i] + "_window"], batch[self.currencies[i] + "_label"]

            output = self.forward(x, i)
            #loss = F.nll_loss(output, y)
            loss += self.cross_entropy_loss[i](output, y)
            
            acc = self.accuracy_score(torch.max(output, dim=1)[1], y)
            self.log(self.currencies[i] +'_train_acc', acc, on_epoch=True, prog_bar=True)

            f1 = self.f1_score(torch.max(output, dim=1)[1], y)
            self.log(self.currencies[i] +'_train_f1', f1, on_epoch=True, prog_bar=True)
        
        loss = loss / torch.tensor(self.num_tasks)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        
        return loss 
    
    def validation_step(self, batch, batch_nb):
        loss = torch.tensor(0.0, device="cuda:0") + torch.tensor(0.0, device="cuda:0")
        
        for i in range(self.num_tasks):
            x, y = batch[self.currencies[i] + "_window"], batch[self.currencies[i] + "_label"]

            output = self(x, i)
            #loss = F.nll_loss(output, y)
            loss += self.cross_entropy_loss[i](output, y)
 
            acc = self.accuracy_score(torch.max(output, dim=1)[1], y)
            self.log(self.currencies[i] +'_val_acc', acc, on_epoch=True, prog_bar=True, reduce_fx=torch.mean)

            f1 = self.f1_score(torch.max(output, dim=1)[1], y)
            self.log(self.currencies[i] +'_val_f1', f1, on_epoch=True, prog_bar=True, reduce_fx=torch.mean)
        
        loss = loss / torch.tensor(self.num_tasks)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_nb):
        loss = torch.tensor(0.0, device="cuda:0") + torch.tensor(0.0, device="cuda:0")
        
        for i in range(self.num_tasks):
            x, y = batch[ self.currencies[i] + "_window"], batch[self.currencies[i] + "_label"]

            output = self(x, i)
#             print(y, torch.max(output, dim=1)[1])
#             print(F.softmax(output)) # mantÄ±ken fark etmiyor
            loss += self.cross_entropy_loss[i](output, y)
            
            acc = self.accuracy_score(torch.max(output, dim=1)[1], y)
            self.log(self.currencies[i] +'_test_acc', acc, on_epoch=True, reduce_fx=torch.mean)

            f1 = self.f1_score(torch.max(output, dim=1)[1], y)
            self.log(self.currencies[i] +'_test_f1', f1, on_epoch=True, reduce_fx=torch.mean)
        
        loss = loss / torch.tensor(self.num_tasks)
        self.log('test_loss', loss, on_epoch=True, reduce_fx=torch.mean)

        
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(model.parameters(), lr= self.learning_rate)#AdamW does weight decay
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
#                                                     step_size=self.scheduler_step, 
#                                                     gamma=self.scheduler_gamma)
        
        self.lr_scheduler = CosineWarmupScheduler(optimizer, 
                                                  warmup=50, 
                                                  max_iters=150* self.train_dl.__len__())
        return [optimizer]#, [{"scheduler": scheduler}]
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step() # Step per iteration
    
    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl
    
def name_model(config):
    task = "multi_task_" + "_".join(config["currency_list"]) if len(config["currency_list"]) > 1 else "single_task_" + config["currency_list"][0]
    classification = "multi_classification" if config["n_classes"] > 2 else "binary_classification"
    lstm = "stack_lstm" if len(config["lstm_hidden_sizes"]) > 1 else "single_lstm"
    #trend_removed = "trend_removed" if config["remove_trend"] else ""
    #loss_weighted = "loss_weighted" if config["loss_weight_calculate"] else ""

    #return "_".join([task, lstm, loss_weighted, classification, trend_removed])
    return "_".join([task, lstm, classification])

CONFIG = {#fix for this project
          "window_size": 50, 
          "dataset_percentages": [0.965, 0.01, 0.025],
          "frenquency": "D", 
          "neutral_quantile": 0.33,
          "batch_size": 64}

config = CONFIG.copy()
config.update({"n_classes": n_classes,
          "currency_list": currency_list,
          "remove_trend": remove_trend,
          "indicators": indicators,
          "imfs": imfs,
          "lstm_hidden_sizes": lstm_hidden_sizes,
          "loss_weight_calculate": loss_weight_calculate,
          "bidirectional": bidirectional})

MODEL_NAME = name_model(config)

CURRENCY_LST = config["currency_list"]
N_CLASSES = config["n_classes"]
LSTM_HIDDEN_SIZES = config["lstm_hidden_sizes"]
BIDIRECTIONAL = config["bidirectional"]
REMOVE_TREND =config["remove_trend"]
INDICATORS = config["indicators"]
IMFS = config["imfs"]
LOSS_WEIGHT_CALCULATE = config["loss_weight_calculate"]

TRAIN_PERCENTAGE, VAL_PERCENTAGE, TEST_PERCENTAGE = config["dataset_percentages"] 
WINDOW_SIZE = config["window_size"]
FREQUENCY = config["frenquency"]
NEUTRAL_QUANTILE = config["neutral_quantile"] if N_CLASSES > 2 else 0 
BATCH_SIZE= config["batch_size"]
#####

X, y, features, dfs = get_data(CURRENCY_LST,
                            N_CLASSES,
                             FREQUENCY, 
                             WINDOW_SIZE,
                             neutral_quantile = NEUTRAL_QUANTILE,
                             log_price=True,
                             remove_trend=REMOVE_TREND,
                             include_indicators = INDICATORS,
                             include_imfs = IMFS
                            )
INPUT_FEATURE_SIZE = X.shape[-1]

train_dataset, val_dataset, test_dataset = [TimeSeriesDataset(CURRENCY_LST, 
                                                          X, 
                                                          y, 
                                                          dtype, 
                                                          TRAIN_PERCENTAGE, 
                                                          VAL_PERCENTAGE, 
                                                          TEST_PERCENTAGE, 
                                                          WINDOW_SIZE) for dtype in ['train', 'val', 'test']]

config["dataset_sizes"] = [len(train_dataset), len(val_dataset), len(test_dataset)]
####
wandb.init(project="price_change_v3",
           config=config,
           name = MODEL_NAME)
logger = WandbLogger()
#     #logger = TensorBoardLogger("../output/models/lstm_model_logs", name="lstm_multi_task")

model = LSTM_based_classification_model(
    train_dataset = train_dataset,
     val_dataset = val_dataset,
     test_dataset = test_dataset,
     calculate_loss_weights = LOSS_WEIGHT_CALCULATE,
     currencies = CURRENCY_LST,
     num_classes = N_CLASSES,
     window_size = WINDOW_SIZE,
     input_size = INPUT_FEATURE_SIZE,
     batch_size=BATCH_SIZE,
     lstm_hidden_sizes = LSTM_HIDDEN_SIZES,
     bidirectional = BIDIRECTIONAL)

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.003,
   patience=20,
   verbose=True,
   mode='min'
)

trainer = pl.Trainer(gpus=-1, 
                     max_epochs= 150,
                     logger = logger, 
                     callbacks=[early_stop_callback])
trainer.fit(model)

trainer.test()
wandb.finish()