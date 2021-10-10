from torch import nn

from .TrainerModule import *

class LSTM_based_classification_model(TrainerModule):
    def __init__(self,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 loss_weightening,
                 currency_list,
                 num_classes,
                 window_size,
                 batch_size,
                 lstm_hidden_size,
                 n_lstm_layers,
                 bidirectional,
                 last_layer_size,
                 dropout_after_each_lstm_layer,
                 batch_norm_after_each_lstm_layer,
                 dropout_before_output_layer,
                 max_epochs = 80,
                 warmup_epoch = 5,
                 learning_rate = 1e-3,
                 weight_decay = 1e-2,
                 random_state = 42,
                 **kwargs
                #  scheduler_step = 10,
                #  scheduler_gamma = 0.1,
                 ):
       
        self.num_classes = num_classes
        self.currency_list = currency_list
        self.num_tasks = len(currency_list)
        self.window_size = train_dataset.seq_len
        self.input_size = train_dataset.x.shape[-1]
        self.batch_size = batch_size
        self.n_lstm_layers = n_lstm_layers
        self.lstm_hidden_sizes = [lstm_hidden_size] * self.n_lstm_layers
        self.bidirectional = bidirectional 
        self.loss_weightening = loss_weightening
        self.dropout_after_each_lstm_layer = dropout_after_each_lstm_layer
        self.batch_norm_after_each_lstm_layer = batch_norm_after_each_lstm_layer
        self.dropout_before_output_layer = dropout_before_output_layer
        self.last_layer_size = last_layer_size

        self.max_epochs = max_epochs
        self.warmup_epoch = warmup_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        #self.scheduler_step = scheduler_step
        #self.scheduler_gamma = scheduler_gamma

        super(LSTM_based_classification_model, self).__init__(train_dataset, val_dataset, test_dataset, random_state)
        
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
     
            lst = [('lstm', lstm_layer)]

            if self.batch_norm_after_each_lstm_layer:
              n_feature = self.lstm_hidden_sizes[i]*2 if self.bidirectional else self.lstm_hidden_sizes[i]   
              batch_norm = nn.BatchNorm2d(num_features=n_feature)
              lst.append(('batch_norm', batch_norm))

            if self.dropout_after_each_lstm_layer:
                dropout = nn.Dropout(self.dropout_after_each_lstm_layer)
                lst.append(('dropout', dropout))
                
            module_dict = nn.ModuleDict(lst)
            
            self.lstm_blocks.append(module_dict)
        
        #task specific layers
        n_feature = self.lstm_hidden_sizes[-1] *2 if bidirectional else self.lstm_hidden_sizes[-1]
        
        self.linear1 =[nn.Linear(n_feature, self.last_layer_size)] * self.num_tasks
        self.linear1 = nn.ModuleList(self.linear1)
        self.activation = nn.ReLU()
        
        if self.dropout_before_output_layer:
          self.dropout1 = nn.Dropout(self.dropout_before_output_layer)
          
        self.output_layers = [nn.Linear(self.last_layer_size, self.num_classes)] * self.num_tasks
        self.output_layers = nn.ModuleList(self.output_layers)
        
        # self.scheduler_step = scheduler_step
        # self.scheduler_gamma = scheduler_gamma
        
    def forward(self, x, n):

        batch_size = x.size()[0]
        
        for i, block in enumerate(self.lstm_blocks):

            if i == 0:
              n_feature = self.input_size 
            else:
              n_feature = self.lstm_hidden_sizes[i-1]*2 if self.bidirectional else self.lstm_hidden_sizes[i-1]   
 
            x = x.view(batch_size, self.window_size, n_feature) #(batch, window_len, feature_size)
            x, _ = block['lstm'](x)
        
            if 'dropout' in block:
                x = block['dropout'](x)
           
            if 'batch_norm' in block:
              x = x.reshape(x.size()[-1], batch_size, self.window_size) #(feature_size, batch, window_len)
              x = block['batch_norm'](x.unsqueeze(0))
  
            if len(x.shape) == 4: #error handling
              x = x.squeeze() 
              
        n_feature = self.lstm_hidden_sizes[-1]*2 if self.bidirectional else self.lstm_hidden_sizes[-1]
        x = x.view(batch_size, self.window_size, n_feature)
        x = x[:, -1, :] # equivalent to return sequence = False on keras :)
        
        x = self.linear1[n](x)
        x = self.activation(x)
        
        if self.dropout_before_output_layer:
            x = self.dropout1(x)
                   
        x = self.output_layers[n](x)

        return x