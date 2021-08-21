import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
import wandb

# from DataPreparation import get_data
from TimeSeriesLearningUtils import TimeSeriesDataset, get_data
from LSTMModel import LSTM_based_classification_model

MAX_EPOCHS = 80
WANDBPROJECT = "deneme"

def name_model(config):
    name =[]
    if config["currency"] >= 0:
        name.append(config[['BTC', 'ETH', 'LTC'][config["currency"]]])
    else:
        name.append("multi_task_" + "_".join(['BTC', 'ETH', 'LTC']))                   
#      if config["currency"] >= 0 else ['BTC', 'ETH', 'LTC']
#     if len(config["currency_list"])  > 1:
#         name.append("multi_task_" + "_".join(config["currency_list"]))
#     else:
#         name.append(config["currency_list"][0])
        
    if config["indicators"] or config["imfs"] or config ["ohlv"]:
        name.append("multi_variate")
    
    lstm = "stack_lstm" if len(config["lstm_hidden_size"]) > 1 else "lstm"
    name.append(lstm)
    
    classification = "multi_clf" if config["n_classes"] > 2 else "binary_clf"
    name.append(classification)
    
    return "_".join(name)
    
config = {"window_size": 100, 
#           "val_percentage": 0.007,
          "val_percentage": 0.023,
          "test_percentage": 0.023,
          "data_frequency": "6h", 
          "pred_frequency": "6h", 
          "ma_period": 10,
          "neutral_quantile": 0.33,
          "batch_size": 128,
          "bidirectional": True, 
          "n_classes": 2,
          "currency": -1,#['BTC'], #['BTC', 'ETH', 'LTC'],
          "remove_trend": True,
          "lstm_hidden_size": 128,
          "stack_lstm": True,
          "loss_weight_calculate": False, 
          "indicators": True, 
          "imfs": False,
          "ohlv": True ,
          "last_layer_fsz": 64,
          "dropout_ratio": 0.5,
          "warmup_epoch": 10,
          "learning_rate": 1e-3,
          "weight_decay": 1e-2}

MODEL_NAME = name_model(config)
wandb.init(project=WANDBPROJECT,
           config=config,
           entity='multi_task_price_prediction',
           name = MODEL_NAME)

config = wandb.config
#####
CURRENCY_LST = ['BTC', 'ETH', 'LTC'][config["currency"]] if config["currency"] >= 0 else ['BTC', 'ETH', 'LTC']
N_CLASSES = config["n_classes"]
LSTM_HIDDEN_SIZES = [config["lstm_hidden_size"]] *3 if config["stack_lstm"] else [config["lstm_hidden_size"]]
BIDIRECTIONAL = config["bidirectional"]
REMOVE_TREND =config["remove_trend"]
LOSS_WEIGHT_CALCULATE = config["loss_weight_calculate"]
VAL_PERCENTAGE =  config["val_percentage"]
TEST_PERCENTAGE = config["test_percentage"] 
DATA_FREQUENCY = config["data_frequency"]
PRED_FREQUENCY = config["pred_frequency"]
MA_PERIOD = config["ma_period"]
WINDOW_SIZE = config["window_size"]
FREQUENCY = config["frenquency"]
NEUTRAL_QUANTILE = config["neutral_quantile"] if N_CLASSES > 2 else 0 
BATCH_SIZE= config["batch_size"]
INDICATORS = config["indicators"]
IMFS = config["imfs"]
OHLV = config["ohlv"]
LAST_LAYER_FSZ = config["last_layer_fsz"]
DROPOUT_RATIO = config["dropout_ratio"]
WARMUP_EPOCH = config["warmup_epoch"]
LEARNING_RATE = config["learning_rate"]
WEIGHT_DECAY = config["weight_decay"]
#####
# X, y, _, _ = get_data(CURRENCY_LST,
#                             N_CLASSES,
#                              FREQUENCY, 
#                              WINDOW_SIZE,
#                              neutral_quantile = NEUTRAL_QUANTILE,
#                              log_price=True,
#                              remove_trend=REMOVE_TREND,
#                              include_indicators = INDICATORS,
#                              include_imfs = IMFS, 
#                              open_high_low_volume = OHLV)
X, y, features, dfs = get_data(currency_lst = CURRENCY_LST,
                               data_frequency =DATA_FREQUENCY,
                               pred_frequency = PRED_FREQUENCY,
                               n_classes = N_CLASSES,
                               window_size=WINDOW_SIZE,
                               neutral_quantile = NEUTRAL_QUANTILE,
                               log_price=True,
                               remove_trend=REMOVE_TREND,
                               ma_period=MA_PERIOD,
                               include_indicators = INDICATORS,
                               include_imfs = IMFS, 
                               open_high_low_volume = OHLV, 
                               drop_missing=True)
                            
INPUT_FEATURE_SIZE = X.shape[-1]

train_dataset, val_dataset, test_dataset = [TimeSeriesDataset(CURRENCY_LST, 
                                                          X, 
                                                          y, 
                                                          dtype, 
                                                          VAL_PERCENTAGE, 
                                                          TEST_PERCENTAGE, 
                                                          WINDOW_SIZE) for dtype in ['train', 'val', 'test']]

config["dataset_sizes"] = [len(train_dataset), len(val_dataset), len(test_dataset)]

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
     bidirectional = BIDIRECTIONAL, 
     last_layer_fsz = LAST_LAYER_FSZ, 
     dropout_ratio = DROPOUT_RATIO, 
     warmup_epoch = WARMUP_EPOCH,
     learning_rate = LEARNING_RATE, 
     weight_decay = WEIGHT_DECAY)

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=3e-3,
   patience=20,
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