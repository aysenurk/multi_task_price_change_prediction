import pandas as pd
import numpy as np
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

from torch.nn import functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger

import wandb

from DataPreparation import get_data

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-cl", "--currency-list", nargs="+", default=[])
parser.add_argument("-trend", default=bool)
parser.add_argument("-weight", default=bool)
parser.add_argument("-classes", default=int)

args = parser.parse_args()
n_classes = int(args.classes)
currency_list = args.currency_list
remove_trend = bool(int(args.trend))
loss_weight_calculate = bool(int(args.weight))

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
    
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self._reset_parameters()
        
        
    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        
        
    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        
        if return_attention:
            return o, attention
        else:
            return o
        
class EncoderBlock(nn.Module):
    
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()
        
        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)
        
        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )
        
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        
        return x
    
class TransformerEncoder(nn.Module):
    
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])
        
    
    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x
    
    
    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model) 
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

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
class TradePredictor(pl.LightningModule): 
    def __init__(self, 
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 calculate_loss_weights,
                 currencies,
                 window_size,
                 batch_size,
                 input_dim, model_dim, num_classes, num_heads, num_layers, 
                 lr,dropout=0.0, input_dropout=0.0,
                ):
        """
        Inputs:
            input_dim - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            num_layers - Number of encoder blocks to use.
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
        """
        super().__init__()
        self.save_hyperparameters()    
        self.num_tasks = len(currencies)

        self._create_model()
        
        self.f1_score = pl.metrics.F1(num_classes=self.hparams.num_classes, average="macro")
        self.accuracy_score = pl.metrics.Accuracy()
        
        self.train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
        self.val_dl = DataLoader(val_dataset, batch_size=batch_size)
        self.test_dl = DataLoader(test_dataset, batch_size=batch_size)
        
        if self.hparams.calculate_loss_weights:
            loss_weights = []
            for i in range(self.num_tasks):
                train_labels = [int(train_dataset[n][self.hparams.currencies[i] +"_label"] )for n in range(train_dataset.__len__())]
                samples_size = pd.DataFrame({"label": train_labels}).groupby("label").size().to_numpy()
                loss_weights.append((1 / samples_size) * sum(samples_size)/2)
            self.weights = loss_weights
        else:
            self.weights = None
            
        if self.weights != None:
            self.cross_entropy_loss = [nn.CrossEntropyLoss(weight= torch.tensor(weights).float()) for weights in self.weights]
        else:
            self.cross_entropy_loss = [nn.CrossEntropyLoss() for _ in range(self.num_tasks)]
        
        self.cross_entropy_loss = torch.nn.ModuleList(self.cross_entropy_loss)
        
    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Linear(self.hparams.input_dim, self.hparams.model_dim)
        
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim)
        # Transformer
        self.transformer = TransformerEncoder(num_layers=self.hparams.num_layers,
                                              input_dim=self.hparams.model_dim,
                                              dim_feedforward=2*self.hparams.model_dim,
                                              num_heads=self.hparams.num_heads,
                                              dropout=self.hparams.dropout)
        # Output classifier per sequence lement
        self.output_net = [nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes)
        ) ]* self.num_tasks
        
        self.output_net = torch.nn.ModuleList(self.output_net)
        
    
    def forward(self, x, i, mask=None, add_positional_encoding=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = x[:,-1,:]
        x = self.output_net[i](x)
        
        
        return x
    
    
    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        
        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(optimizer, 
                                                  warmup=50, 
                                                  max_iters = 80* self.train_dl.__len__())
        return optimizer
    
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step() # Step per iteration

    def _calculate_loss(self, batch, mode="train"):
        
        loss = (torch.tensor(0.0, device="cuda:0", requires_grad=True) + \
                torch.tensor(0.0, device="cuda:0", requires_grad=True)) 
        for i in range(self.num_tasks):
            x, y = batch[self.hparams.currencies[i] + "_window"], batch[self.hparams.currencies[i] + "_label"]
  
            preds = self.forward(x, i, add_positional_encoding=True) # No positional encodings as it is a set, not a sequence!

            loss += self.cross_entropy_loss[i](preds, y) # Softmax/CE over set dimension
            
            acc = self.accuracy_score(torch.max(preds, dim=1)[1], y)
            self.log(self.hparams.currencies[i] + "_%s_acc" % mode, acc, on_step=False,  prog_bar=True, on_epoch=True)
            
            f1 = self.f1_score(torch.max(preds, dim=1)[1], y)
            self.log(self.hparams.currencies[i] + "_%s_f1" % mode, f1, on_step=False,  prog_bar=True, on_epoch=True)
        
        loss = loss / torch.tensor(self.num_tasks)
        self.log("%s_loss" % mode, loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")
    
    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test") 
    
    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl
def name_model(config):
    task = "multi_task_" + "_".join(config["currency_list"]) if len(config["currency_list"]) > 1 else "single_task_" + config["currency_list"][0]
    classification = "multi_classification" if config["n_classes"] > 2 else "binary_classification"
    trend_removed = "trend_removed" if config["remove_trend"] else ""
    loss_weighted = "loss_weighted" if config["loss_weight_calculate"] else ""

    return "_".join([task, "multi_head_attention", loss_weighted, classification, trend_removed])
    
CONFIG = {#fix for this project
          "window_size": 50, 
          "dataset_percentages": [0.97, 0.007, 0.023],
          "frenquency": "D", 
          "neutral_quantile": 0.33,
          "batch_size": 16,}

config = CONFIG.copy()
config.update({"n_classes": n_classes,
          "currency_list": currency_list,
          "remove_trend": remove_trend,
          "loss_weight_calculate": loss_weight_calculate})

MODEL_NAME = name_model(config)
####
CURRENCY_LST = config["currency_list"]
N_CLASSES = config["n_classes"]
REMOVE_TREND =config["remove_trend"]
LOSS_WEIGHT_CALCULATE = config["loss_weight_calculate"]
###
#FIXED
TRAIN_PERCENTAGE, VAL_PERCENTAGE, TEST_PERCENTAGE = config["dataset_percentages"] 
WINDOW_SIZE = config["window_size"]
FREQUENCY = config["frenquency"]
NEUTRAL_QUANTILE = config["neutral_quantile"] if N_CLASSES > 2 else 0 
BATCH_SIZE= config["batch_size"]

X, y, features, dfs = get_data(CURRENCY_LST,
                                N_CLASSES,
                                 FREQUENCY, 
                                 WINDOW_SIZE,
                                 neutral_quantile = NEUTRAL_QUANTILE,
                                 log_price=True,
                                 remove_trend=True,
                                 include_indicators = False,
                                 include_imfs = False
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

wandb.init(project="price_change_2",
           config=config,
           name = MODEL_NAME)
logger = WandbLogger()

model = TradePredictor( 
                    input_dim=INPUT_FEATURE_SIZE,
                    model_dim=64,
                    num_heads=8,
                    num_classes=N_CLASSES,
                    num_layers=4,
                    dropout=0.5,
                    lr=5e-4,
                    train_dataset = train_dataset,
                    val_dataset = val_dataset,
                    test_dataset = test_dataset,
                    calculate_loss_weights = LOSS_WEIGHT_CALCULATE,
                    currencies = CURRENCY_LST,
                    window_size = WINDOW_SIZE,
                    batch_size=BATCH_SIZE,)

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.003,
   patience=25,
   verbose=True,
   mode='min'
)
trainer = pl.Trainer(#default_root_dir=root_dir, 
                     #checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                     gpus=-1 , 
                     max_epochs=80,
                     gradient_clip_val=2,
                     progress_bar_refresh_rate=1, 
                     logger = logger, 
                     callbacks=[early_stop_callback])

trainer.fit(model)
trainer.test()
wandb.finish()