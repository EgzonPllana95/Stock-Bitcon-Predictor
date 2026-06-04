import torch
import numpy as np
from datetime import datetime, timedelta
from data_loader import DataLoader
from model import build_model

class Predictor:
    """Handels loading a tranied model and making predictions on new data"""
    def __init__(self, model_path = "models/lstm_model.pt"):
        #Loades a trained model from disk
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #These gets filled when load() is called
        self.model = None
        self.scaler = None
        self.config = None
        #Lodes after thr Predictor is created
        self.load()
    
    def load(self):
        """Loads the weight, scaler and config of models from disk"""
        #With weights_only = False allows loading the scaler as an object not just tensors
        #Otherwise Pytorch weights_only is default True for security
        checkpoint = torch.load(self.model_path, map_location = self.device, weights_only = False)
        self.scaler = checkpoint["scaler"]
        self.config = checkpoint["config"]
        #Rebuulding the architecture using saved config, matches the architeture used during traning
        #otherwise the weights wont fit
        self.model = build_model(
            input_size = self.config["input_size"],
            hidden_size = self.config["hidden_size"],
            num_layers = self.config["num_layers"],
            dropout = self.config["dropout"]
        ).to(self.device)
        #Loades the traind weights into the empty model
        self.model.load_state_dict(checkpoint["model_state_dict"])
        #Setting it to evaluation mode nad disables dropout for consistent predictions
        self.model.eval()
        print(f"Loaded model trained on {self.config['ticker']}")
    
    def get_latest_sequence(self, ticker = None):
        """
        Gets the most recent days of price data, ready to feed into the model.
        ticker:  If None, uses the same ticker the model was trained on.
        Returns: tensor shaped (1, 60, 1) — one sequence of 60 days
        """
        #Default to the ticker the model was trained on
        if ticker is None:
            ticker = self.config["ticker"]
        seq_len = self.config["sequence_length"]
        #Fetch a bit more than seq_len days to be safe
        #(weekends and holidays mean fewer trading days than calendar days)
        start_date = (datetime.today() - timedelta(days=seq_len * 2)).strftime("%Y-%m-%d")
        loader = DataLoader(
            ticker = ticker,
            start_date = start_date,
            sequence_length = seq_len
        )
        loader.fetch()
        loader.scale_prices()
        #Take ONLY the last seq_len days — that's what the model expects as input
        latest = loader.scaled_prices[-seq_len:]
        #Reshape to (1, sequence_length, 1) — model expects a batch
        #1 = batch size of one (we're predicting just one sequence)
        latest_tensor = torch.FloatTensor(latest).unsqueeze(0).to(self.device)
        return latest_tensor, loader.scaler

    def predict_next_day(self, ticker = None):
        """
        Predicts tomorrow's closing price.
        ticker:  If None, uses the same ticker the model was trained on.
        Returns: predicted price in real dollars (e.g. 187.42)
        """
        #Get the latest sequence (and the fresh scaler from current data)
        latest_tensor, fresh_scaler = self.get_latest_sequence(ticker)
        #torch.no_grad() — we're predicting, not training, so skip gradient tracking
        with torch.no_grad():
            scaled_prediction = self.model(latest_tensor)
        #Move back to CPU and convert to numpy for the scaler
        scaled_value = scaled_prediction.cpu().numpy()
        #Convert from [0, 1] back to real dollars
        #We use the FRESH scaler from current data (not the training scaler)
        #because price ranges may have shifted since training
        real_price = fresh_scaler.inverse_transform(scaled_value)
        return float(real_price[0][0])

    def predict_multi_day(self, days = 7, ticker = None):
        """
        Predicts several days ahead by feeding each prediction back into the model.
        Heads up: errors add up over time — day 7 is much less reliable than day 1.
        days:    How many days into the future to predict
        ticker:  If None, uses the same ticker the model was trained on.
        Returns: list of predicted prices in real dollars
        """
        latest_tensor, fresh_scaler = self.get_latest_sequence(ticker)
        #Make a copy we can modify (the original is needed below for the scaler)
        current_sequence = latest_tensor.clone()
        scaled_predictions = list()
        with torch.no_grad():
            for _ in range(days):
                # Predict next value
                next_scaled = self.model(current_sequence)
                scaled_predictions.append(next_scaled.cpu().numpy()[0][0])
                #Slide the window: drop oldest day, append predicted day
                #current_sequence shape: (1, 60, 1)
                #next_scaled shape: (1, 1) — needs reshape to (1, 1, 1)
                next_reshaped = next_scaled.unsqueeze(-1)
                #Concatenate along the time dimension (dim=1), skipping the oldest day
                current_sequence = torch.cat(
                    [current_sequence[:, 1:, :], next_reshaped],
                    dim = 1
                )
        # onvert all scaled predictions back to real prices
        scaled_array = np.array(scaled_predictions).reshape(-1, 1)
        real_prices = fresh_scaler.inverse_transform(scaled_array)
        return real_prices.flatten().tolist()

if __name__ == "__main__":
    # Quick test — run with: python src/predict.py
    predictor = Predictor(model_path="models/lstm_model.pt")
    # --- Single-day prediction ---
    print("\n--- Next-day prediction ---")
    next_price = predictor.predict_next_day()
    print(f"Predicted next-day price for {predictor.config['ticker']}: ${next_price:.2f}")
    # --- Multi-day prediction ---
    print("\n--- 7-day forecast ---")
    forecast = predictor.predict_multi_day(days=7)
    for i, price in enumerate(forecast, start=1):
        print(f"Day +{i}: ${price:.2f}")