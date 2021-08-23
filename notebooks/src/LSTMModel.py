import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

from .TimeSeriesLearningUtils import CosineWarmupScheduler

MAX_EPOCHS = 80

class LSTM_based_classification_model(pl.LightningModule):
    def __init__(self,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 calculate_loss_weights,
                 currency_list,
                 num_classes,
                 window_size,
                 batch_size,
                 lstm_hidden_size,
                 n_lstm_layers,
                 bidirectional,
                 last_layer_fsz,
                 dropout_after_each_lstm_layer,
                 dropout_before_output_layer,
                 warmup_epoch = 5,
                 learning_rate = 1e-3,
                 weight_decay = 1e-2,
                 **kwargs
                #  scheduler_step = 10,
                #  scheduler_gamma = 0.1,
                 ):
        
        super().__init__()
        self.num_classes = num_classes
        self.currency_list = currency_list
        self.num_tasks = len(currency_list)
        self.window_size = window_size
        self.input_size = train_dataset.x.shape[-1]
        self.batch_size = batch_size
        self.n_lstm_layers = n_lstm_layers
        self.lstm_hidden_sizes = [lstm_hidden_size] * self.n_lstm_layers
        self.bidirectional = bidirectional 
        self.loss_weightening = calculate_loss_weights
        self.dropout_after_each_lstm_layer = dropout_after_each_lstm_layer
        self.dropout_before_output_layer = dropout_before_output_layer
        self.last_layer_fsz = last_layer_fsz
        self.learning_rate = learning_rate
        self.warmup_epoch = warmup_epoch
        self.weight_decay = weight_decay
        
        if calculate_loss_weights:
            loss_weights = []
            for i in range(self.num_tasks):
                train_labels = [int(train_dataset[n][self.currency_list[i] +"_label"] )for n in range(train_dataset.__len__())]
                samples_size = pd.DataFrame({"label": train_labels}).groupby("label").size().to_numpy()
                loss_weights.append((1 / samples_size) * sum(samples_size)/2)
            self.weights = loss_weights
        else:
            self.weights = None
        
        # self.lstm_1 = nn.LSTM(input_size = self.input_size, 
        #                       num_layers=1, 
        #                       batch_first=True, 
        #                       hidden_size = self.lstm_hidden_sizes[0], 
        #                       bidirectional = bidirectional)
        # self.batch_norm1 = nn.BatchNorm2d(num_features=self.lstm_hidden_sizes[0]*2 if bidirectional else self.lstm_hidden_sizes[0])
        
        # if len(self.lstm_hidden_sizes) > 1:
        #     self.lstm_2 = nn.LSTM(input_size = self.lstm_hidden_sizes[0] *2 if bidirectional else self.lstm_hidden_sizes[0], 
        #                           num_layers=1, 
        #                           batch_first=True, 
        #                           hidden_size = self.lstm_hidden_sizes[1], 
        #                           bidirectional = bidirectional)
        #     self.batch_norm2 = nn.BatchNorm2d(num_features=self.lstm_hidden_sizes[1]*2 if bidirectional else self.lstm_hidden_sizes[1])

        #     self.lstm_3 = nn.LSTM(input_size = self.lstm_hidden_sizes[1]*2 if bidirectional else self.lstm_hidden_sizes[1], 
        #                           num_layers=1, 
        #                           batch_first=True, 
        #                           hidden_size = self.lstm_hidden_sizes[2], 
        #                           bidirectional = bidirectional)
        #     self.batch_norm3 = nn.BatchNorm2d(num_features=self.lstm_hidden_sizes[2]*2 if bidirectional else self.lstm_hidden_sizes[2])
        
        
        # self.dropout = nn.Dropout(self.dropout_ratio)
        
        self.lstm_blocks = nn.ModuleList()
        
        for i in range(self.n_lstm_layers):

            if i == 0:
              input_size = self.input_size 
            else:
              input_size = self.lstm_hidden_sizes[i-1]*2 if self.bidirectional else self.lstm_hidden_sizes[i-1]   
            
            lstm_layer = nn.LSTM(input_size = input_size, 
                                  num_layers=1, 
                                  batch_first=True, 
                                  hidden_size = self.lstm_hidden_sizes[i], 
                                  bidirectional = self.bidirectional)
            
            n_feature = self.lstm_hidden_sizes[i]*2 if self.bidirectional else self.lstm_hidden_sizes[i]   
            batch_norm = nn.BatchNorm2d(num_features=n_feature)
            lst = [('lstm', lstm_layer), ('batch_norm', batch_norm)]
  
            if self.dropout_after_each_lstm_layer:
                dropout = nn.Dropout(self.dropout_after_each_lstm_layer)
                lst.append(('dropout', dropout))
                
            module_dict = nn.ModuleDict(lst)
            
            self.lstm_blocks.append(module_dict)
        
        n_feature = self.lstm_hidden_sizes[-1] *2 if bidirectional else self.lstm_hidden_sizes[-1]
        
        self.linear1 =[nn.Linear(n_feature, self.last_layer_fsz)] * self.num_tasks
        self.linear1 = torch.nn.ModuleList(self.linear1)
        self.activation = nn.ReLU()
        
        if self.dropout_before_output_layer:
          self.dropout1 = nn.Dropout(self.dropout_before_output_layer)
          
        self.output_layers = [nn.Linear(self.last_layer_fsz, self.num_classes)] * self.num_tasks
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
        
        # self.scheduler_step = scheduler_step
        # self.scheduler_gamma = scheduler_gamma
        
    def forward(self, x, n):

        batch_size = x.size()[0]
        
        # x = x.view(batch_size, self.window_size, self.input_size) #(batch, window_len, feature_size)
        # x, _  = self.lstm_1(x)
        
        # x = self.dropout(x)

        # x = x.reshape(x.size()[-1], batch_size, self.window_size) #(feature_size, batch, window_len)
        # x = self.batch_norm1(x.unsqueeze(0))
        
        # if len(self.lstm_hidden_sizes) > 1:
            
        #     x = x.view(batch_size, self.window_size, x.size()[1])
        #     x, _  = self.lstm_2(x)

        #     x = self.dropout(x)

        #     x = x.reshape(x.size()[-1], batch_size, self.window_size) #(feature_size, batch, window_len)
        #     x = self.batch_norm2(x.unsqueeze(0))

        #     x = x.view(batch_size, self.window_size, x.size()[1])
        #     x, _  = self.lstm_3(x)

        #     x = self.dropout(x)

        #     x = x.reshape(x.size()[-1], batch_size, self.window_size) #(feature_size, batch, window_len)
        #     x = self.batch_norm3(x.unsqueeze(0))
        
        for i, block in enumerate(self.lstm_blocks):

            if i == 0:
              n_feature = self.input_size 
            else:
              n_feature = self.lstm_hidden_sizes[i-1]*2 if self.bidirectional else self.lstm_hidden_sizes[i-1]   
 
            x = x.view(batch_size, self.window_size, n_feature) #(batch, window_len, feature_size)
            x, _ = block['lstm'](x)
        
            if 'dropout' in block:
                x = block['dropout'](x)
           
            x = x.reshape(x.size()[-1], batch_size, self.window_size) #(feature_size, batch, window_len)
        
            x = block['batch_norm'](x.unsqueeze(0))
  
            if len(x.shape) == 4: #error handling
              x = x.squeeze() 
              
        #x = x.view(batch_size, self.window_size, x.size()[1])
        n_feature = self.lstm_hidden_sizes[-1]*2 if self.bidirectional else self.lstm_hidden_sizes[-1]
        x = x.view(batch_size, self.window_size, n_feature)
        x = x[:, -1, :] # equivalent to return sequence = False on keras :)
        
        #x = self.dropout(x)
        
        x = self.linear1[n](x)
        x = self.activation(x)
        
        if self.dropout_before_output_layer:
            x = self.dropout1(x)
                   
        output = self.output_layers[n](x)
    
        return output
    
    def step(self, batch, step_type = 'train'):
        loss = (torch.tensor(0.0, device="cuda:0", requires_grad=True) + \
                torch.tensor(0.0, device="cuda:0", requires_grad=True)) 
        
        # araştırılabilir
        for i in range(self.num_tasks):
            x, y = batch[self.currency_list[i] + "_window"], batch[self.currency_list[i] + "_label"]

            output = self.forward(x, i)
            #loss = F.nll_loss(output, y)
            loss += self.cross_entropy_loss[i](output, y)
            
            acc = self.accuracy_score(torch.max(output, dim=1)[1], y)
            self.log(f"{self.currency_list[i]}_{step_type}_acc", acc, on_epoch=True, prog_bar=True)

            f1 = self.f1_score(torch.max(output, dim=1)[1], y)
            self.log(f"{self.currency_list[i]}_{step_type}_f1", f1, on_epoch=True, prog_bar=True)
        
        loss = loss / torch.tensor(self.num_tasks)
        self.log(f"{step_type}_loss", loss, on_epoch=True, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr= self.learning_rate, 
                                      weight_decay=self.weight_decay)

#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
#                                                     step_size=self.scheduler_step, 
#                                                     gamma=self.scheduler_gamma)
        
        self.lr_scheduler = CosineWarmupScheduler(optimizer, 
                                                  warmup = self.train_dl.__len__() * self.warmup_epoch, 
                                                  max_iters = MAX_EPOCHS * self.train_dl.__len__())
        return [optimizer]#, [{"scheduler": scheduler}]
    
    def training_step(self, batch, batch_nb):
        
        loss = self.step(batch, "train")
        
        return loss 
    
    def validation_step(self, batch, batch_nb):
        
        self.step(batch, "val")
    
    def test_step(self, batch, batch_nb):
        
        self.step(batch, "test")
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step() # Step per iteration
    
    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl