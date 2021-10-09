import time 
import torch
import torchmetrics
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb

from src.TimeSeriesLearningUtils import *
from src.LSTMModel import *
from src.TransformerEncoderModel import *
from src.MLP import *
MODEL_CLASSES = {'lstm': LSTM_based_classification_model, 'transformer': TradePredictor, 'mlp':MLP}


def name_model(config):
    name =[]
    
    if len(config["currency_list"])  > 1:
        name.append("multi_task_" + "_".join(config["currency_list"]))
    else:
        name.append(config["currency_list"][0])
        
    if config["indicators"] or config["imfs"] or config ["ohlv"] or config['decompose']:
        name.append("multi_variate")

    if config['model_name'] == 'lstm':
      lstm = "stack_lstm" if config["n_lstm_layers"] > 1 else "lstm"
      name.append(lstm)
    else:
      name.append(config['model_name'])
    
    name.append(config["pred_frequency"])
    classification = "multi_clf" if config["num_classes"] > 2 else "binary_clf"
    name.append(classification)
    
    return "_".join(name)


def experiment(config, seed=42):    
    X, y, _, _, _ = get_data(**config)
    train_dataset, val_dataset, test_dataset = [TimeSeriesDataset(X,
                                                                  y,
                                                                  dtype, 
                                                                  **config) for dtype in ['train', 'val', 'test']]

    config["dataset_sizes"] = [len(train_dataset), len(val_dataset), len(test_dataset)]
    WANDBPROJECT = "furkan-deneme9"
    MODEL_NAME = name_model(config)
    
    wandb.init(project=WANDBPROJECT,######
               config=config,
               #entity='multi_task_price_prediction',
               name = MODEL_NAME )

    logger = WandbLogger()

    model = MODEL_CLASSES[config['model_name']](train_dataset = train_dataset,
                                                val_dataset = val_dataset,
                                                test_dataset = test_dataset,
                                                random_state = seed,
                                                **config)

    early_stop_callback = EarlyStopping(
       monitor='val_loss',
       min_delta=0.003,
       patience=20,
       verbose=True,
       mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='../output/',
        filename = MODEL_NAME + str(time.time()) +'-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(gpus=-1, 
                         max_epochs= config['max_epochs'],
                         logger = logger, 
                         callbacks=[early_stop_callback, checkpoint_callback])
    trainer.fit(model)

    trainer.test(ckpt_path = checkpoint_callback.best_model_path)

    wandb.finish()


from sklearn.model_selection import ParameterGrid

data_setting = {
    "num_classes": [2],#[2,3],
    "currency_list":  [['ETH']],#,[['BTC']],#[['ADA'], ['BTC'], ['ETH'], ['LTC'], ['XRP']],
    "window_size": [100], #test 
    "dataset_percentages": [[0.9, 0.05, 0.05]],
    "data_frequency": ["1d"], 
    "pred_frequency": ["1d"], 
    "ma_period": [7],
    "neutral_quantile": [0.33],
    "log_price": [True],
    "indicators": [False],#[True, False],
    "imfs": [False], 
    "remove_trend": [True],
    "decompose": [False],
    "ohlv": [False]
    }

model_params = { 
    "model_name":['mlp'],
    "lstm_hidden_size": [128],#[64, 128],
    "n_lstm_layers": [2], #[1,3,5],
    "bidirectional": [True],
    "dropout_after_each_lstm_layer": [0.5],
    "dropout_before_output_layer": [0.5],
    "batch_norm_after_each_lstm_layer":[True],
    "last_layer_size": [32]
    }

hparams = {
    "loss_weightening": [True],
    "batch_size": [8],
    "max_epochs":[30],
    "warmup_epoch": [10],
    "learning_rate": [1e-3],
    "weight_decay": [1e-2]
    }

param_grid = {**model_params, **hparams, **data_setting}


from pprint import pprint

if __name__ == '__main__':

    for i in range(1):
        for config in (ParameterGrid(param_grid)):
            #pprint(config)
            seed = 42#i
            experiment(config, seed)
