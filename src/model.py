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

    def forward(self, x):
        """
        Defains how the data flows thorugh the model
        Pytorch will automatically call when output = model(x)
        """
        #Need initial memory state since LSTM requiers that, start both at 0
        batch_size = x.size(0)
        #Short term memory
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        #Long term memory
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        #All hidden layers for each time step ( btach, seq_length, hidden_size)
        lstm_out, _ = self.lstm(x, (h0, c0))
        #Last timestep is taken and the model "thinks" after seeing all 60 days
        last_output = lstm_out[:, -1, :]
        #Applies dropout
        out = self.dropout(last_output)
        #Fully connected layers to predtion
        out = self.fc(out)
        return out
    
    