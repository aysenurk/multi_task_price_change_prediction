import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import WandbLogger
import wandb

from src.TimeSeriesLearningUtils import *
from src.LSTMModel import *

MAX_EPOCHS = 80
WANDBPROJECT = "deneme"

def name_model(config):
    name =[]
    if len(config["currency_list"])  > 1:
        name.append("multi_task_" + "_".join(config["currency_list"]))
    else:
        name.append(config["currency_list"][0])
        
    if config["indicators"] or config["imfs"] or config ["ohlv"]:
        name.append("multi_variate")
    
    lstm = "stack_lstm" if config["n_lstm_layers"] > 1 else "lstm"
    name.append(lstm)
    
    name.append(config["pred_frequency"])
    classification = "multi_clf" if config["num_classes"] > 2 else "binary_clf"
    name.append(classification)
    
    return "_".join(name)
    
config = {
          "log_price": True,
          "dataset_percentages": [0.96, 0.02, 0.02],
          "data_frequency": "1d", 
          "pred_frequency": "D",
          "num_classes": 3,
          "currency_list": ['BTC', 'ETH', 'LTC'], #['LTC'],# 
          
          "window_size": 50, 
          "ma_period": 10, 
          "neutral_quantile": 0.33,
          "remove_trend": True,
          "indicators": True, 
          "imfs": False,
          "ohlv": False,
          
          "bidirectional": True, 
          "dropout_after_each_lstm_layer": 0.5,
          "dropout_before_output_layer": 0.5,
          "lstm_hidden_size": 128,
          "n_lstm_layers": 3,
          "calculate_loss_weights": True, 
          "last_layer_fsz": 128,
          
          "batch_size": 64,
          "warmup_epoch": 10,
          "learning_rate": 1e-3,
          "weight_decay": 1e-2,
          }

MODEL_NAME = name_model(config)
wandb.init(project=WANDBPROJECT,
           config=config,
           entity='multi_task_price_prediction',
           name = MODEL_NAME)

config = wandb.config

X, y, features, dfs = get_data(**config)

train_dataset, val_dataset, test_dataset = [TimeSeriesDataset(X,
                                                              y,
                                                              dtype, 
                                                              **config) for dtype in ['train', 'val', 'test']]

config["dataset_sizes"] = [len(train_dataset), len(val_dataset), len(test_dataset)]

logger = WandbLogger()

model = LSTM_based_classification_model(train_dataset = train_dataset,
                                        val_dataset = val_dataset,
                                        test_dataset = test_dataset,
                                        **config)

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=3e-3,
   patience=15,
   verbose=True,
   mode='min'
)
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='../../output/',
    filename= MODEL_NAME +'-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
)

trainer = pl.Trainer(gpus=-1, 
                     max_epochs= MAX_EPOCHS,
                     logger = logger, 
                     callbacks=[early_stop_callback, checkpoint_callback])
trainer.fit(model)

trainer.test(ckpt_path = checkpoint_callback.best_model_path)
wandb.finish()