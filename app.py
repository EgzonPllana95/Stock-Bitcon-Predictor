import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os
import sys

# Add src/ to the Python path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import DataLoader
from train import Trainer
from predict import Predictor
from tickers import TickerRegistry
from news_advisor import fetch_news, get_recommendation


# --- Page setup ---
st.set_page_config(
    page_title="Stock & Crypto Predictor",
    page_icon="📈",
    layout="wide"
)

# --- Title ---
st.title("📈 Stock & Crypto Predictor")
st.markdown("LSTM-based price prediction using PyTorch and Yahoo Finance data.")


# --- Sidebar ---
st.sidebar.header("Settings")

# Initialize the ticker registry (local shortcut list)
registry = TickerRegistry()

# Single search field — works with names, tickers, and partial matches
user_input = st.sidebar.text_input(
    "Enter a Stock or Crypto",
    placeholder="e.g. Apple, Axfood, Bitcoin, AAPL",
    help="Type a company name or ticker — searches Yahoo Finance"
)

ticker = None

if user_input:
    query = user_input.strip()

    # Step 1: Search the local registry first (fast)
    local_matches = registry.search(query)

    # Step 2: Also search Yahoo Finance directly (broader coverage)
    with st.spinner("Searching..."):
        yahoo_matches = TickerRegistry.search_yahoo(query, max_results=5)

    # Combine results — local first, then Yahoo
    all_matches = {**local_matches, **yahoo_matches}

    if all_matches:
        # If only one match, use it directly
        if len(all_matches) == 1:
            ticker = list(all_matches.values())[0]
            label = list(all_matches.keys())[0]
            st.sidebar.success(f"✓ Found: **{label}** ({ticker})")
        else:
            # Multiple matches — let user pick
            chosen = st.sidebar.selectbox(
                "Pick the right one:",
                options=list(all_matches.keys()),
                index=0
            )
            ticker = all_matches[chosen]
            st.sidebar.success(f"✓ Selected: **{ticker}**")
    else:
        # Last resort: try the input as a direct ticker
        candidate = query.upper()
        if TickerRegistry.validate(candidate):
            ticker = candidate
            st.sidebar.success(f"✓ Valid ticker: **{ticker}**")
        else:
            st.sidebar.error(f"❌ Could not find '{query}'. Try a company name or ticker.")

# Forecast slider
st.sidebar.markdown("---")
forecast_days = st.sidebar.slider(
    "How many days to forecast?",
    min_value=1,
    max_value=30,
    value=7,
    help="How many days into the future to predict"
)

# Single action button
st.sidebar.markdown("---")
predict_button = st.sidebar.button("📊 Show Forecast", use_container_width=True)


# --- Helper functions ---

def fetch_history(ticker, days=365):
    """Fetches recent price history for plotting."""
    from datetime import datetime, timedelta
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")

    loader = DataLoader(ticker=ticker, start_date=start, sequence_length=60)
    df = loader.fetch()
    return df


def get_currency_symbol(ticker):
    """
    Frågar yfinance vilken valuta tickern handlas i och returnerar
    rätt symbol. Faller tillbaka på själva valutakoden om symbolen
    är okänd, och på tom sträng om yfinance inte vet (då visas bara
    siffran utan symbol — bättre än en FELAKTIG symbol).

    Detta löser felet där svenska aktier (t.ex. SWED-A.ST, priser i
    SEK) visades med "$" framför, vilket är fel valuta på rätt siffra.
    """
    import yfinance as yf
    try:
        info = yf.Ticker(ticker).info
        code = info.get("currency", "")
    except Exception:
        code = ""

    symbols = {
        "USD": "$",
        "SEK": "kr",
        "EUR": "€",
        "GBP": "£",
        "NOK": "kr",
        "DKK": "kr",
    }
    # Saknas koden i tabellen: visa koden som text (t.ex. "327.70 SEK")
    # — korrekt om än mindre snyggt. Hellre sant än vilseledande.
    return symbols.get(code, (code + " ") if code else "")


def fmt_price(value, symbol):
    """
    Formaterar ett pris med rätt valutasymbol. För symboler som
    står EFTER beloppet i svensk konvention (kr) sätter vi den
    efter; för $/€/£ före. Detta är bara kosmetik — siffran är
    densamma oavsett.
    """
    if symbol in ("kr", "kr "):
        return f"{value:.2f} kr"
    return f"{symbol}{value:.2f}"


def plot_prices(df, predictions=None, ticker="", currency_label="USD"):
    """Builds a Plotly chart with historical prices and (optional) predictions."""
    fig = go.Figure()

    # Historical prices
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"].values.flatten(),
        mode="lines",
        name="Historical Price",
        line=dict(color="#1f77b4", width=2)
    ))

    # Predictions (if provided)
    if predictions is not None:
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(predictions))

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode="lines+markers",
            name="Predicted Price",
            line=dict(color="#ff7f0e", width=2, dash="dash")
        ))

    fig.update_layout(
        title=f"{ticker} — Price History" + (" + Forecast" if predictions is not None else ""),
        xaxis_title="Date",
        # Y-axeln visar nu RÄTT valuta i stället för hårdkodat "USD".
        yaxis_title=f"Price ({currency_label})",
        height=500,
        hovermode="x unified",
        template="plotly_white"
    )

    return fig


def get_or_train_model(ticker):
    """
    Returns the path to a trained model for this ticker.
    If no model exists, trains one automatically.
    """
    model_path = f"models/{ticker.replace('-', '_').replace('.', '_').replace('^', '')}.pt"

    # If a model already exists, just use it
    if os.path.exists(model_path):
        return model_path

    # Otherwise, train one with default settings
    with st.spinner(f"Training a new model for {ticker}... This takes 1-2 minutes (only the first time)."):
        trainer = Trainer(
            ticker=ticker,
            epochs=20,
            sequence_length=60,
            model_save_path=model_path
        )
        trainer.run()

    return model_path


# --- Session state init ---
# Initiera minnesnycklar EN gång. 'if ... not in' gör att vi bara
# sätter dem första gången — annars nollställs minnet vid varje klick.
if "forecast" not in st.session_state:
    st.session_state.forecast = None
if "forecast_ticker" not in st.session_state:
    st.session_state.forecast_ticker = None
if "forecast_days_used" not in st.session_state:
    st.session_state.forecast_days_used = None
if "advice" not in st.session_state:
    st.session_state.advice = None


# --- Main area ---

# Stop early if no valid ticker
if ticker is None:
    st.info("👈 Enter a stock or crypto name in the sidebar to get started.")
    st.stop()

# Show selected ticker
st.subheader(f"Selected: {ticker}")

# Hämta valutan EN gång för den här tickern. Används både för
# metrikerna, grafens y-axel och prognos-siffrorna nedan.
import yfinance as yf
try:
    _currency_code = yf.Ticker(ticker).info.get("currency", "") or "USD"
except Exception:
    _currency_code = "USD"
currency_symbol = get_currency_symbol(ticker)

# --- Run prediction when user clicks the button ---
# Själva BERÄKNINGEN ligger kvar i knapp-blocket. VISNINGEN sker
# utifrån session_state längre ner så den överlever omkörningar.
if predict_button:
    try:
        # Get or train the model (silent — user just sees a single spinner)
        model_path = get_or_train_model(ticker)

        # Make predictions
        with st.spinner("Generating forecast..."):
            predictor = Predictor(model_path=model_path)
            forecast = predictor.predict_multi_day(days=forecast_days, ticker=ticker)

        st.session_state.forecast = forecast
        st.session_state.forecast_ticker = ticker
        st.session_state.forecast_days_used = forecast_days

    except Exception as e:
        st.error(f"Forecast failed: {e}")

# Finns en giltig sparad prognos för PRECIS den ticker som visas nu?
has_forecast = (
    st.session_state.forecast is not None
    and st.session_state.forecast_ticker == ticker
)

# Always show the historical chart
try:
    df = fetch_history(ticker, days=365)

    # Price metrics — nu med RÄTT valutasymbol via fmt_price().
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Price", fmt_price(df['Close'].iloc[-1].item(), currency_symbol))
    col2.metric("365-day High", fmt_price(df['Close'].max().item(), currency_symbol))
    col3.metric("365-day Low", fmt_price(df['Close'].min().item(), currency_symbol))

    # Rita grafen utifrån MINNET, inte utifrån knappen.
    chart_placeholder = st.empty()

    if has_forecast:
        chart_placeholder.plotly_chart(
            plot_prices(
                df,
                predictions=st.session_state.forecast,
                ticker=ticker,
                currency_label=_currency_code,
            ),
            use_container_width=True,
        )
    else:
        chart_placeholder.plotly_chart(
            plot_prices(df, ticker=ticker, currency_label=_currency_code),
            use_container_width=True,
        )

except Exception as e:
    st.error(f"Could not fetch data for {ticker}: {e}")
    st.stop()


# --- Forecast metrics + table ---
# Ritas utifrån session_state, så det överlever omkörningar precis
# som grafen.
if has_forecast:
    forecast = st.session_state.forecast
    days_used = st.session_state.forecast_days_used

    # Show forecast metrics
    st.subheader(f"📊 {days_used}-Day Forecast")

    col1, col2, col3 = st.columns(3)
    last_price = df["Close"].iloc[-1].item()
    next_price = forecast[0]
    change_pct = ((next_price - last_price) / last_price) * 100

    col1.metric("Current Price", fmt_price(last_price, currency_symbol))
    col2.metric(
        "Tomorrow's Prediction",
        fmt_price(next_price, currency_symbol),
        delta=f"{change_pct:+.2f}%",
    )
    col3.metric(
        f"Day +{days_used} Prediction",
        fmt_price(forecast[-1], currency_symbol),
    )

    # Forecast table — samma valutasymbol i tabellen.
    forecast_df = pd.DataFrame({
        "Day": [f"+{i+1}" for i in range(len(forecast))],
        "Predicted Price": [fmt_price(p, currency_symbol) for p in forecast]
    })
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    # Disclaimer
    st.info("""
    **Disclaimer:** Stock prices are influenced by many factors not captured in
    historical price data alone (news, earnings, market sentiment, etc).
    These predictions are for educational purposes — not financial advice.
    """)


# --- Nyheter + AI-bedömning ---
# Ligger efter grafen så df garanterat finns (annars har koden
# redan kört st.stop() ovan).

st.subheader("Senaste nyheter")

news = fetch_news(ticker)

if not news:
    st.info("Inga nyheter hittades för denna ticker.")
else:
    for n in news:
        st.markdown(
            f"**[{n['title']}]({n['link']})**  \n_{n['publisher']}_"
        )

st.subheader("AI-bedömning")

language = st.radio(
    "Språk för bedömning",
    options=["Svenska", "English"],
    horizontal=True,
)

lang_code = "en" if language == "English" else "sv"

# Knapp så att Gemini bara anropas på begäran (varje anrop kostar kvot).
# Svaret sparas i session_state så det överlever omkörningar.
if st.button("Generera AI-bedömning"):
    with st.spinner("Analyserar..."):
        recent_prices = df["Close"].tail(30).values.flatten().tolist()
        st.session_state.advice = get_recommendation(
            ticker, news, recent_prices, language=lang_code
        )

if st.session_state.advice:
    st.text(st.session_state.advice)

# Disclaimer — byter språk så hela UI:t är konsekvent.
if lang_code == "en":
    st.warning(
        "AI-generated assessments are "
        "NOT financial advice and must not be used for investment decisions."
    )
else:
    st.warning(
        "AI-genererade bedömningar är "
        "INTE finansiell rådgivning och ska inte användas för investeringsbeslut."
    )


# --- Footer ---
st.markdown("---")
st.caption("Built with PyTorch, Streamlit, Plotly, and Yahoo Finance via yfinance.")