import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(ticker, start_date, end_date):
    print(f"Downloading data for {ticker}...")
    # auto_adjust=True handles stock splits/dividends automatically
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    df = df.dropna()
    return df

def backtest(df, short_window=50, long_window=200):
    """
    Implements a Vectorized Backtest (No 'For' Loops).
    Strategy: Golden Cross (Buy when Short SMA > Long SMA).
    """
    
    # 1. Calculate Indicators
    # Rolling windows are optimized in Pandas (underlying C code)
    df['SMA_50'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_200'] = df['Close'].rolling(window=long_window).mean()
    
    # 2. GENERATE SIGNALS (The "Vectorized" Flex)
    # Instead of iterating row-by-row (slow), we use np.where to check
    # the condition on the entire memory block at once.
    # Signal = 1.0 (Long/Buy) or 0.0 (Cash/Flat)
    df['Signal'] = np.where(df['SMA_50'] > df['SMA_200'], 1.0, 0.0)
    
    # 3. CALCULATE RETURNS
    # pct_change() calculates the daily market return
    df['Market_Return'] = df['Close'].pct_change()
    
    # CRITICAL: Shift the signal by 1 day.
    # We calculate the signal using Today's Close data.
    # We cannot trade at Today's Close (market is shut). We trade Tomorrow Open.
    # If we don't shift, we are "cheating" by using future data.
    df['Strategy_Return'] = df['Market_Return'] * df['Signal'].shift(1)
    
    return df.dropna()

if __name__ == "__main__":
    symbol = "SPY"
    # Fetch 5 years of data
    raw_data = fetch_data(symbol, "2020-01-01", "2025-01-01")
    
    # Run the engine
    results = backtest(raw_data)
    
    # Validation: Print the tail to show the strategy is working
    print("\n--- BACKTEST RESULTS (Last 5 Days) ---")
    print(results[['Close', 'SMA_50', 'SMA_200', 'Signal', 'Strategy_Return']].tail())
    
    # Quick sanity check sum (Total Return uncompounded)
    print(f"\nSimple Total Return: {results['Strategy_Return'].sum():.2%}")