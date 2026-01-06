import sys
import os
import joblib # Dùng để lưu Scaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import các module đã viết
from src.config import TIME_STEP, EPOCH, BATCH_SIZE, LR, FEATURE_COLUMNS
from src.data_ingest import get_realtime_data, add_technical_indicators
from src.preprocess import prepare_multivariate_data
from src.model import build_cnn_lstm_model

def train_pipeline(stock_symbol='AAPL'):

    # kéo data real time về
    # Lấy 5 năm dữ liệu 
    try:
        df_raw = get_realtime_data(stock_symbol, years=5)
        print(f"   -> Đã tải {len(df_raw)} dòng dữ liệu.")
    except Exception as e:
        return

    # feature engineering (thêm RSI, MACD, Volume)
    df_rich = add_technical_indicators(df_raw)
    print(f"   -> Các đặc trưng hiện có: {list(df_rich.columns)}")

    # BƯỚC 3: scaling,window
    X, y, scaler_y, scaler_X = prepare_multivariate_data(df_rich, TIME_STEP)
    
    # tự động lấy số lượng features (6)
    n_features = X.shape[2]
    print(f"   -> Input Shape: {X.shape}") 
    print(f"   -> Time Step: {TIME_STEP}, Features: {n_features}")

    # chia tập Train/Test
    # shuffle=False dữ liệu chuỗi thời gian (không được xáo trộn ngày tháng)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, shuffle=False)
    print(f"   -> Train set: {X_train.shape} (65%)")
    print(f"   -> Test set:  {X_test.shape} (35%)")

    # build,train
    model = build_cnn_lstm_model(TIME_STEP, n_features, LR)
    
    # Callback: Lưu model tốt nhất và dừng sớm nếu không học thêm được
    checkpoint = ModelCheckpoint(f"models/{stock_symbol}_best_model.h5", save_best_only=True, monitor='val_loss', mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCH,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    # lưu Model cuối cùng
    model.save(f"models/{stock_symbol}_final_model.h5")
    
    #lưu Scaler của y (giá Close) để sau này dự báo ngược ra tiền thật
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(scaler_y, f"models/{stock_symbol}_scaler_y.pkl")
    
if __name__ == "__main__":
    TICKERS = ['AAPL', 'TSLA']
    
    for ticker in TICKERS:
        train_pipeline(ticker)
        
