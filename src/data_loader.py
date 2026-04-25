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
            end_date = datetime.today().strftime("%Y-%m-&d")
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
    
    