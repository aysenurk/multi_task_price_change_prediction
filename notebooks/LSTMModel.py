import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

from TimeSeriesLearningUtils import TimeSeriesDataset, CosineWarmupScheduler

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
        self.batch_norm1 = nn.BatchNorm2d(num_features=self.lstm_hidden_sizes[0])
        
        if len(self.lstm_hidden_sizes) > 1:
            self.lstm_2 = nn.LSTM(input_size = self.lstm_hidden_sizes[0], 
                                  num_layers=1, 
                                  batch_first=True, 
                                  hidden_size = self.lstm_hidden_sizes[1], 
                                  bidirectional = bidirectional)
            self.batch_norm2 = nn.BatchNorm2d(num_features=self.lstm_hidden_sizes[1])

            self.lstm_3 = nn.LSTM(input_size = self.lstm_hidden_sizes[1], 
                                  num_layers=1, 
                                  batch_first=True, 
                                  hidden_size = self.lstm_hidden_sizes[2], 
                                  bidirectional = bidirectional)
            self.batch_norm3 = nn.BatchNorm2d(num_features=self.lstm_hidden_sizes[2])
        
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
        # araştırılabilir
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
#             print(F.softmax(output)) # mantıken fark etmiyor
            loss += self.cross_entropy_loss[i](output, y)
            
            acc = self.accuracy_score(torch.max(output, dim=1)[1], y)
            self.log(self.currencies[i] +'_test_acc', acc, on_epoch=True, reduce_fx=torch.mean)

            f1 = self.f1_score(torch.max(output, dim=1)[1], y)
            self.log(self.currencies[i] +'_test_f1', f1, on_epoch=True, reduce_fx=torch.mean)
        
        loss = loss / torch.tensor(self.num_tasks)
        self.log('test_loss', loss, on_epoch=True, reduce_fx=torch.mean)

        
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr= self.learning_rate)#AdamW does weight decay
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
#                                                     step_size=self.scheduler_step, 
#                                                     gamma=self.scheduler_gamma)
        
        self.lr_scheduler = CosineWarmupScheduler(optimizer, 
                                                  warmup=self.train_dl.__len__() * 10, 
                                                  max_iters = 80 * self.train_dl.__len__())
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