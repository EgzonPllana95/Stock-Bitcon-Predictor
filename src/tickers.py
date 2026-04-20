class TickerRegistry:
    """Known stock and crypto tickers that will be organized by category"""

    TICKERS = {
        "US Stocks": {
            "Apple": "AAPL",
            "Microsoft": "MSFT",
            "Google (Alphabet)": "GOOGL",
            "Amazon": "AMZN",
            "Tesla": "TSLA",
            "Nvidia": "NVDA",
            "Meta (Facebook)": "META",
            "Netflix": "NFLX",
            "AMD": "AMD",
            "Intel": "INTC",
            "Coca-Cola": "KO",
            "Disney": "DIS",
            "Visa": "V",
            "JPMorgan Chase": "JPM",
            "Berkshire Hathaway": "BRK-B",
        },

        "Swedish Stocks": {
            "Volvo B": "VOLV-B.ST",
            "Ericsson B": "ERIC-B.ST",
            "H&M B": "HM-B.ST",
            "Investor B": "INVE-B.ST",
            "Atlas Copco A": "ATCO-A.ST",
            "SEB A": "SEB-A.ST",
            "Nordea": "NDA-SE.ST",
            "Handelsbanken A": "SHB-A.ST",
            "Swedbank A": "SWED-A.ST",
            "Evolution": "EVO.ST",
            "Sinch": "SINCH.ST",
            "Telia": "TELIA.ST",
            "SKF B": "SKF-B.ST",
            "Sandvik": "SAND.ST",
            "ABB": "ABB.ST",
        },

        "Crypto": {
            "Bitcoin": "BTC-USD",
            "Ethereum": "ETH-USD",
            "Solana": "SOL-USD",
            "Cardano": "ADA-USD",
            "Dogecoin": "DOGE-USD",
            "Ripple (XRP)": "XRP-USD",
            "Polkadot": "DOT-USD",
            "Litecoin": "LTC-USD",
            "Chainlink": "LINK-USD",
            "Polygon": "MATIC-USD",
        },

        "Indices": {
            "S&P 500": "^GSPC",
            "Nasdaq 100": "^NDX",
            "Dow Jones": "^DJI",
            "OMX Stockholm 30": "^OMX",
            "FTSE 100 (London)": "^FTSE",
            "DAX (Germany)": "^GDAXI",
            "Nikkei 225 (Japan)": "^N225",
        },

        "European Stocks": {
            "Nokia (Finland)": "NOKIA.HE",
            "Novo Nordisk (Denmark)": "NOVO-B.CO",
            "Equinor (Norway)": "EQNR.OL",
            "SAP (Germany)": "SAP.DE",
            "LVMH (France)": "MC.PA",
            "ASML (Netherlands)": "ASML.AS",
            "Barclays (UK)": "BARC.L",
        },
    }

    def __init__(self, custom_tickers = None):
        """
        Creates a new TickerRegistry
        You can pass your own ticker list, or leave it empty to use the default one
        """
        #Provides dict or falls back to class TickkerTegistry default
        self.tickers = custom_tickers if custom_tickers else self.TICKERS

    def get_all(self) -> dict:
        """
        Returns all tickers as one flat dict.
        The 'Category — Name' format makes them searchable in dropdowns
        """
        flat = dict()
        for category, tickers in self.tickers.items():
            for name, symbol in tickers.items():
                flat[f"{category} - {name}"] = symbol
        return flat
    
    def get_categories(self) -> list:
        """Returns a list of category names (["Swe Stocks", Crypto,,,])"""
        return list(self.tickers.keys())
    
    def get_by_category(self, category: str) -> dict:
        """
        Returns all tickers in one category ('Swe Stocks')
        Returns an empty dict if the category doesn't exist."""
        return self.tickers.get(category, dict())
    
    def search(self, query: str) -> dict:
        """
        Searches tickers by name, symbol, or category
        Case-insensitive and matches partial words ('volvo' finds Volvo B)
        """
        query = query.lower().strip()
        #Query thar are empty returns evertyhing, its for initial UI
        if not query:
            return self.get_all()
        results = dict()
        for category, tickers in self.tickers.items():
            for name, symbol in tickers.items():
                #Combines the string to one searchable one so it matches all the fields
                searchable = f"{category} {name} {symbol}".lower()
                if query in searchable:
                    results[f"{category} - {name}"] = symbol
        return results
    
    @staticmethod
    def validate(ticker: str) -> bool:
        """
        Checks if a ticker exists on Yahoo Finance
        @staticmethod means you can call it without creating an instance: TickerRegistry.validate('AAPL')
        """
        import yfinance as yf
        try:
            data = yf.Ticker(ticker).history(period="1d")
            return not data.empty
        except Exception:
            return False

    def __len__(self) -> int:
        """Lets you use len(registry) to get the total number of tickers."""
        return sum(len(tickers) for tickers in self.tickers.values())

    def __repr__(self) -> str:
        """Controls what gets printed when you do print(registry). Useful for debugging."""
        return f"TickerRegistry(categories = {len(self.tickers)}, total_tickers = {len(self)})"


if __name__ == "__main__":
    # Quick test — run with: python src/tickers.py
    registry = TickerRegistry()
    print(registry)  # Uses __repr__
    print(f"Total tickers: {len(registry)}\n")  # Uses __len__
    #Categories
    print("Available categories:")
    for category in registry.get_categories():
        print(f"- {category}")
    #Sample of all tickers
    print("\nSample of all tickers:")
    all_tickers = registry.get_all()
    for name, symbol in list(all_tickers.items())[:5]:
        print(f"{name}  →  {symbol}")
    #Search tests
    print("\nSearch tests:")
    print("\n  Searching for 'volvo':")
    for name, symbol in registry.search("volvo").items():
        print(f"{name}  →  {symbol}")
    print("\n  Searching for 'btc':")
    for name, symbol in registry.search("btc").items():
        print(f"{name}  →  {symbol}")
    print("\n  Searching for 'swedish':")
    for name, symbol in registry.search("swedish").items():
        print(f"{name}  →  {symbol}")
    # --- Validation tests (static method — no instance needed) ---
    print("\nValidation tests (requires internet):")
    print(f"  AAPL valid? {TickerRegistry.validate('AAPL')}")
    print(f"  VOLV-B.ST valid? {TickerRegistry.validate('VOLV-B.ST')}")
    print(f"  FAKETICKER123? {TickerRegistry.validate('FAKETICKER123')}")