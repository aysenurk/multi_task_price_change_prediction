from torch import nn

from .TrainerModule import *

class MLP(TrainerModule):
    
    def __init__(self,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 loss_weightening,
                 currency_list,
                 num_classes,
                 window_size,
                 batch_size,
                 last_layer_size,
                 dropout_before_output_layer,
                 max_epochs = 80,
                 warmup_epoch = 5,
                 learning_rate = 1e-3,
                 weight_decay = 1e-2,
                # scheduler_step = 10,
                # scheduler_gamma = 0.1,
                random_state = 42,
                 **kwargs
                 ):
        
        self.num_classes = num_classes
        self.currency_list = currency_list
        self.num_tasks = len(currency_list)
        self.window_size = window_size
        self.input_size = train_dataset.x.shape[-1]
        self.batch_size = batch_size
        self.loss_weightening = loss_weightening
        self.dropout_before_output_layer = dropout_before_output_layer
        self.last_layer_size = last_layer_size

        self.max_epochs = max_epochs
        self.warmup_epoch = warmup_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        #self.scheduler_step = scheduler_step
        #self.scheduler_gamma = scheduler_gamma

        super(MLP, self).__init__(train_dataset, val_dataset, test_dataset, random_state)

        self.layers = nn.Sequential(
          nn.Linear(self.window_size*self.input_size, 100),
          nn.ReLU(),
          #nn.BatchNorm1d(100),
          #nn.Dropout(0.50),
          #nn.Linear(100, 100),
          #nn.ReLU(),
          #nn.BatchNorm1d(100),
          #nn.Dropout(0.50),
          #nn.Linear(50, self.num_classes)
          #nn.Linear(100, 50)
        )

        #task specific layers
        self.linear1 =[nn.Linear(100, self.last_layer_size)] * self.num_tasks
        self.linear1 = nn.ModuleList(self.linear1)
        self.activation = nn.ReLU()
        
        if self.dropout_before_output_layer:
          self.dropout1 = nn.Dropout(self.dropout_before_output_layer)
          
        self.output_layers = [nn.Linear(self.last_layer_size, self.num_classes)] * self.num_tasks
        self.output_layers = nn.ModuleList(self.output_layers)
    
    
    def forward(self, x, n):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = self.linear1[n](x)
        x = self.activation(x)   
        if self.dropout_before_output_layer:
            x = self.dropout1(x)       
        x = self.output_layers[n](x)
        return x