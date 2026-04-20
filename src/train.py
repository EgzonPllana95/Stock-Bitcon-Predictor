import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import numpy as np
import os
from data_loader import fetch_stock_data, prepare_sequances
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
                 model_save_path = "models"):
        pass