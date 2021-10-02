import torch
import torchmetrics
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .TimeSeriesLearningUtils import *

class TrainerModule(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, random_state, **kwargs):
        super(TrainerModule, self).__init__()
        pl.seed_everything(random_state)

        if self.loss_weightening:
            loss_weights = []
            for i in range(self.num_tasks):
                train_labels = [int(train_dataset[n][self.currency_list[i] +"_label"] )for n in range(len(train_dataset))]
                samples_size = pd.DataFrame({"label": train_labels}).groupby("label").size().to_numpy()
                loss_weights.append((1 / samples_size) * sum(samples_size)/2)
            self.weights = loss_weights
        else:
            self.weights = None

        if self.weights != None:
            self.cross_entropy_loss = [nn.CrossEntropyLoss(weight= torch.tensor(weights).float()) for weights in self.weights]
        else:
            self.cross_entropy_loss = [nn.CrossEntropyLoss()] * self.num_tasks
        
        self.cross_entropy_loss = nn.ModuleList(self.cross_entropy_loss)
        
        self.f1_score = torchmetrics.F1(num_classes=self.num_classes, average="macro")
        self.accuracy_score = torchmetrics.Accuracy()
  
        self.train_dl = DataLoader(train_dataset, batch_size=self.batch_size, shuffle = True)
        self.val_dl = DataLoader(val_dataset, batch_size=self.batch_size)
        self.test_dl = DataLoader(test_dataset, batch_size=self.batch_size)


    def step(self, batch, step_type = 'train'):
        loss = (torch.tensor(0.0, device="cuda:0", requires_grad=True) + torch.tensor(0.0, device="cuda:0", requires_grad=True)) 
        accuracy_sum = (torch.tensor(0.0, device="cuda:0", requires_grad=False) + torch.tensor(0.0, device="cuda:0", requires_grad=False)) 
        
        for i in range(self.num_tasks):
            x, y = batch[self.currency_list[i] + "_window"], batch[self.currency_list[i] + "_label"]

            output = self.forward(x, i)
            loss += self.cross_entropy_loss[i](output, y)
            
            acc = self.accuracy_score(torch.max(output, dim=1)[1], y)
            accuracy_sum += acc
            
            self.log(f"{self.currency_list[i]}_{step_type}_acc", acc, on_epoch=True, prog_bar=True)
            
            f1 = self.f1_score(torch.max(output, dim=1)[1], y)
            self.log(f"{self.currency_list[i]}_{step_type}_f1", f1, on_epoch=True, prog_bar=True)
        
        loss = loss / torch.tensor(self.num_tasks)
        avg_acc = accuracy_sum / torch.tensor(self.num_tasks)
        self.log(f"{step_type}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{step_type}_acc", avg_acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr= self.learning_rate, 
                                      weight_decay=self.weight_decay)

#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
#                                                     step_size=self.scheduler_step, 
#                                                     gamma=self.scheduler_gamma)
        
        self.lr_scheduler = CosineWarmupScheduler(optimizer, 
                                                  warmup = len(self.train_dl) * self.warmup_epoch, 
                                                  max_iters =  len(self.train_dl) * self.max_epochs)
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