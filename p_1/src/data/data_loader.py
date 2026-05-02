import pandas as pd
import yfinance as yf

def download_prices(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """
    Downloads adjusted closing prices from Yahoo Finance.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker, for example:
        - AAPL
        - MSFT
        - ^GSPC
        - BTC-USD
        - GC=F

    start_date:
        Initial date in YYYY-MM-DD format.

    end_date:
        Final date in YYYY-MM-DD format.

    Returns
    -------
    prices:
        A pandas Series with daily closing prices.
    """
    data = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            "No data was downloaded. Check the ticker symbol or the selected dates."
        )

    # yfinance usually returns a simple DataFrame for one ticker.
    # However, depending on the version, it may return MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):
        if ("Close", ticker) in data.columns:
            prices = data[("Close", ticker)]
        elif ("Adj Close", ticker) in data.columns:
            prices = data[("Adj Close", ticker)]
        else:
            # Fallback: try to get the first close-like column
            close_columns = [
                col for col in data.columns
                if "Close" in str(col)
            ]
            if not close_columns:
                raise ValueError("Could not identify a closing price column.")
            prices = data[close_columns[0]]
    else:
        if "Close" in data.columns:
            prices = data["Close"]
        elif "Adj Close" in data.columns:
            prices = data["Adj Close"]
        else:
            raise ValueError("Could not identify a closing price column.")

    prices = prices.dropna()
    prices.name = ticker

    return prices
