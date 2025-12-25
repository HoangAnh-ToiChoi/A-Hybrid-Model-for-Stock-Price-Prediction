import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import os
from config import TIME_STEP  

def get_latest_stock_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", progress=False)      
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)            
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)            
        return df
    except Exception as e:
        print(f"Lỗi tải dữ liệu: {e}")
        return None

# Hàm 2: Chuẩn bị dữ liệu cho Model dự đoán
def prepare_input_for_prediction(ticker, df):
    
    #1. Load Scaler cũ
    #2. Lấy 100 ngày cuối cùng
    #3. Scale và Reshape (1, 100, 1)
    # Đường dẫn đến scaler
    scaler_path = f"data/processed/{ticker}/scaler.pkl"
    
    # Kiểm tra xem file scaler có tồn tại không
    if not os.path.exists(scaler_path):
        return None, None, f"Không tìm thấy file Scaler tại {scaler_path}. Hãy chạy src/save_scalers.py trước!"

    try:
        # Load Scaler
        scaler = joblib.load(scaler_path)
        
        # Chỉ lấy cột Close
        data = df[['Close']].values
        
        # Kiểm tra đủ dữ liệu không
        if len(data) < TIME_STEP:
            return None, scaler, f"Dữ liệu không đủ {TIME_STEP} ngày (chỉ có {len(data)})"
            
        # Lấy đúng 100 ngày cuối cùng để dự đoán ngày mai
        last_data = data[-TIME_STEP:]
        
        # Scale dữ liệu (Chỉ transform, KHÔNG fit)
        scaled_data = scaler.transform(last_data)
        
        # Reshape thành 3D [1 mẫu, 100 ngày, 1 đặc trưng]
        X_input = scaled_data.reshape(1, TIME_STEP, 1)
        
        return X_input, scaler, None
        
    except Exception as e:
        return None, None, f"Lỗi xử lý dữ liệu: {e}"