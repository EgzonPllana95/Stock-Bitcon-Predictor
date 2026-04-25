import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

class DataLoader:
    """
    Will handel all data operations for hte LSTM pipeline
    It will download the price history from Yahoo finance
    Scales the price of a range 0,1
    Creates sliding-window sequences for LSTM traning
    """

    def __init__(self,
                 ticker = "AAPL",
                 start_date = "2015-01-01",
                 end_date = None,
                 sequence_length = 60):
        """
        Sets up a DataLoader for one stock or crypto.
            - ticker: Yahoo Finance symbol (e.g. "AAPL", "BTC-USD", "VOLV-B.ST")
            - start_date: Where to start the data, format "YYYY-MM-DD"
            - end_date: Where to stop. If None, today's date is used.
            - sequence_length: How many days of history the model looks at (60 = ~3 months)
        """
        #Uses todays date if a end_date is not given
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        #When fetch() and prepare_sequence() is called they will get filled
        #Orignial df (DataFrame) from Yahoo
        self.raw_data = None
        #Preproccesin with MinMaxScaler vor converting prices back
        self.scaler = None
        #Scaled to 0,1
        self.scaled_prices = None

    def fetch(self):
        """Downloads historical price data from Yahoo Finance that returns with pandas Dataframe"""
        print(f"Fetching {self.ticker} from {self.start_date} to {self.end_date}...")
        #yf (YahooFinance) downloads and returns a pandas Dataframe
        df = yf.download(
            self.ticker,
            start = self.start_date,
            end = self.end_date,
            #Hides the progressing bar for a more cleaner output
            progress = False
        )
        #Validation for the ticker if its valid or not
        if df.empty:
            raise ValueError(f"No data found for ticker {self.ticker}! Be sure to typ the correct symbol")
        self.raw_data = df
        print(f"Donloaded {len(df)} days of data")
        return df
    
    def scale_prices(self):
        """
        Scales closing prices to a [0, 1] range
        Neural networks train better with small, consistent numbers
        Raw prices like $150 make training unstable.
        """

        #Makes sure that we fetch the data first
        if self.raw_data is None:
            raise RuntimeError("Call fetch() before scale_prices()!")
        #Reshaping to 2d since MinMaxScaler requiers 2D input
        prices = self.raw_data["Close"].values.reshape(-1, 1)
        #Using MinMaxScaler to scale the prices from yf data
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        self.scaled_prices = self.scaler.fit_transform(prices)
        return self.scaled_prices
    
    def create_sequences(self):
        """
        Creates sliding window sequences for LSTM traning
        X = prices from day (N) to day (N+59)
        y = price on day (N+60) 
        """
        #Validation that the price is scaled first
        if self.scaled_prices is None:
            raise RuntimeError("Call scale_prices() before create_sequences().")
        X, y = list(), list()
        #Slides a window across the data that starts att sequences_length
        for i in range(self.sequence_length, len(self.scaled_prices)):
            #Takes previous 60 days
            X.append(self.scaled_prices[i - self.sequence_length:i])
            #Targeted one is the price on day [i], after the window
            y.append(self.scaled_prices[i])
        #Convertingthe X, y to arrays, needed for Pytorch
        X = np.array(X)
        y = np.array(y)
        print(f"Created {len(X)} sequences of length {self.sequence_length}")
        return X, y
    
    def run(self):
        """
        Convenience method:
            - fetches
            - scales
            - creates sequanes in one call
        """
        self.fetch()
        self.scale_prices()
        return self.create_sequences()

if __name__ == "__main__":
    loader = DataLoader(ticker="AAPL", start_date="2020-01-01", sequence_length=60)
    # Run the full pipeline
    X, y = loader.run()
    print("\n--- Result ---")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"\nFirst sequence (first 5 days):")
    print(X[0][:5].flatten())
    print(f"Target for first sequence: {y[0][0]:.4f}")