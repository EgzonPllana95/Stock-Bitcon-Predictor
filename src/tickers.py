class TickkerTegistry:
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
        self.tickers = custom_tickers if custom_tickers else self.TICKERS

        