import os
import yfinance as yf
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def fetch_news(ticker_symbol: str, max_items: int = 5) -> list[dict]:
    """
    Hämtar de senaste nyheterna för en given ticker via yfinance.

    - ticker_symbol: t.ex. "AAPL" eller "BTC-USD"
    - max_items: hur många nyheter vi vill ha tillbaka (default 5)

    Returnerar en lista med dictar: {title, publisher, link, published}.
    Returnerar tom lista om inga nyheter finns.
    """
    ticker = yf.Ticker(ticker_symbol)

    # .news kan vara tom eller saknas för mindre kända tickers.
    raw_news = getattr(ticker, "news", []) or []

    cleaned = []
    for item in raw_news[:max_items]:
        # yfinance har ändrat format mellan versioner. Nyare versioner
        # lägger datan under "content". Vi hanterar båda fallen.
        content = item.get("content", item)

        title = content.get("title", "Ingen rubrik")
        publisher = (
            content.get("provider", {}).get("displayName")
            if isinstance(content.get("provider"), dict)
            else content.get("publisher", "Okänd källa")
        )
        link = (
            content.get("canonicalUrl", {}).get("url")
            if isinstance(content.get("canonicalUrl"), dict)
            else content.get("link", "")
        )
        published = content.get(
            "pubDate", content.get("providerPublishTime", "")
        )

        cleaned.append({
            "title": title,
            "publisher": publisher,
            "link": link,
            "published": published,
        })

    return cleaned


def get_recommendation(
    ticker_symbol: str,
    news_items: list[dict],
    recent_prices: list[float],
    language: str = "sv",
) -> str:
    """
    Skickar nyheter + prisrörelse till Gemini och ber om en
    strukturerad bedömning. Returnerar svaret som text.
    """
    api_key = os.environ.get("API_KEY")
    if not api_key:
        return "FEL: API_KEY saknas i .env-filen."

    # NYTT: i google-genai skapar man en Client med nyckeln.
    # (I gamla paketet hette det genai.configure() — nu en klient.)
    client = genai.Client(api_key=api_key)

    # news_text och price_text byggs EXAKT som förut.
    if news_items:
        news_text = "\n".join(
            f"- {n['title']} ({n['publisher']})" for n in news_items
        )
    else:
        news_text = "Inga nyheter tillgängliga."

    if recent_prices:
        first = recent_prices[0]
        last = recent_prices[-1]
        change_pct = ((last - first) / first) * 100
        price_text = (
            f"Priset har gått från {first:.2f} till {last:.2f} "
            f"({change_pct:+.1f}%) under perioden."
        )
    else:
        price_text = "Ingen prisdata tillgänglig."

    # Samma format-mallar som förut, helt oförändrade.
    if language == "en":
        system_prompt = (
            "You are a financial analyst providing an educational "
            "assessment for an academic project. You NEVER give binding "
            "financial advice. Always answer in EXACTLY this format:\n\n"
            "RECOMMENDATION: [BUY / HOLD / SELL]\n"
            "REASONING: [4-6 sentences]\n"
            "RISK: [2-3 sentences]\n"
            "DISCLAIMER: This is not financial advice."
        )
    else:  # "sv"
        system_prompt = (
            "Du är en finansanalytiker som ger en utbildningsmässig "
            "bedömning för ett akademiskt projekt. Du ger ALDRIG bindande "
            "finansiella råd. Svara ALLTID i exakt detta format:\n\n"
            "REKOMMENDATION: [KÖP / BEHÅLL / SÄLJ]\n"
            "MOTIVERING: [4-6 meningar]\n"
            "RISK: [2-3 meningar]\n"
            "DISCLAIMER: Detta är inte finansiell rådgivning."
        )

    user_message = (
        f"Aktie/krypto: {ticker_symbol}\n\n"
        f"Prisutveckling:\n{price_text}\n\n"
        f"Senaste nyheter:\n{news_text}\n\n"
        f"Ge din bedömning enligt formatet."
    )

    # NYTT: det nya paketet HAR ett separat system-fält igen
    # (via config), så vi behöver inte slå ihop som i det gamla.
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",   # från din list_models-körning
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=1000,
        ),
    )

    # Svaret ligger fortfarande i .text — samma som förut.
    return response.text