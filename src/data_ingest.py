# src/data_ingest.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_realtime_data(symbol, years=5):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    
    df = yf.download(symbol, start=start_date, end=end_date)
    
    # xử lý multiIndex 
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df

def add_technical_indicators(df):
    data = df.copy()
    
    # thay thế sma20,60 bằng sma10s,50
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    #rsi
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    #macd
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    
    #volume
    data['Volume'] = data['Volume'].fillna(0)
    
    data.dropna(inplace=True)
    
    return data