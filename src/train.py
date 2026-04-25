import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from datetime import datetime
import os
from data_loader import DataLoader
from model import build_model

class Trainer:
    """Handels the full traning pipeline for the LSTM stock predition model"""

    def __init__(self,
                 ticker = "AAPL",
                 start_date = "2020-01-01",
                 end_date = None,
                 sequence_length = 60,
                 batch_size = 32,
                 epochs = 20,
                 learning_rate = 0.001,
                 hidden_size = 64,
                 num_layers = 2,
                 dropout = 0.2,
                 model_save_path = "models/lstm_model.pt"):
        """
        Sets up a Trainer with all the settings needed for training
        ticker: Stock symbol from Yahoo Finance (e.g. "AAPL" for Apple)
        start_date: When the price data starts ("YYYY-MM-DD")
        end_date: When it ends. If None, today's date is used automatically.
        sequence_length: How many past days the model looks at to predict the next day (60)
        batch_size: How many examples the model sees before learning from them (32)
        epochs: How many times the model goes through all the data (20)
        learning_rate: How big steps the model takes when learning (0.001 is a safe default)
        hidden_size: How many "memory cells" each LSTM layer has (more = smarter, but slower)
        num_layers: How many LSTM layers stacked on top of each other (2 is standard)
        dropout: How many neurons are randomly turned off during training to prevent overfitting (20%)
        model_save_path:  Where the trained model gets saved on your computer
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_save_path = model_save_path
        #Gets filled upp later in by other methods
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = None
        self.model = None
        self.history = {"train_loss": [], "val_loss": []}
        print(f"Trainer initialized. Using device: {self.device}")

    def prepare_data(self):
        """Prepares stock data and creates train/validations sets"""
        #Uses Dataloader class för fetch, sacle and create sequences
        loader = DataLoader(
            ticker = self.ticker,
            start_date = self.start_date,
            end_date = self.end_date,
            sequence_length = self.sequence_length
        )
        X, y = loader.run()
        #Saves the scaler for later perpess
        self.scaler = loader.scaler
        #Spliting the traning to 80% traning and 20% validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        #Converting them to arrays with numpy because of PyTorch
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        #Handels batching and shuffling during traning
        train_dataset = TensorDataset(X_train, y_train)
        self.train_loader = TorchDataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        #Validation on data
        self.X_val = X_val
        self.y_val = y_val
        print(f"Train: {len(X_train)} sequences | Val: {len(X_val)} sequences")
    
    def build(self):
        """Builds the LSTM model, loss function, and optimizer."""
        self.model = build_model(
            input_size = 1,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            dropout = self.dropout
        ).to(self.device)
        #MSE Mean Squared Error losses
        self.criterion = nn.MSELoss()
        #Using Adam as our optimizer, default choice
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        print(self.model)

    