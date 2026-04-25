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