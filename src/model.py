import torch
import torch.nn as nn

class LSTModel:
    """Model that will predict stock prize"""
    def __init__(self,
                 input_size = 1,
                 hidden_size = 64,
                 num_layers = 2,
                 dropout = 0.2):
        #Always most be called apon får Pytorch models
        #It initislisersnn.Module tat keeps tracks of all parameters
        super(LSTModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #Heart of the model
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            #Inout-tensor will have the formela (batch_size, sequence_length, input_size)
            #Exempel: (32, 26, 1), 32 = stocks, 26 = days, 1 = featuers
            batch_first = True,
            #Dropout between layers only if we have more then 1 layer
            dropout = dropout if num_layers > 1 else 0
        )
        #Dropout layers
        self.dropout = nn.Dropout(dropout)
        #Linear prediction
        self.fc = nn.Linear(hidden_size, 1)

    
