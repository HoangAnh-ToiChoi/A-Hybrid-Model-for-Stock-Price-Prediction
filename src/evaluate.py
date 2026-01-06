import sys
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Thêm đường dẫn để import được module trong folder src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import các hàm từ hệ thống mới
from src.config import TIME_STEP, DATA_PATH
from src.data_ingest import get_realtime_data, add_technical_indicators
from src.preprocess import prepare_multivariate_data

# Tắt GPU trên Mac để tránh lỗi 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def evaluate_model(ticker):

    #tải data real time
    try:
        df_raw = get_realtime_data(ticker, years=5)
        df_rich = add_technical_indicators(df_raw)
    except Exception as e:
        return

    # tiền xử lí data
    X, y, _, _ = prepare_multivariate_data(df_rich, TIME_STEP)
    
    # 3. load model & scaler đã lưu
    model_path = f"models/{ticker}_final_model.h5"
    scaler_path = f"models/{ticker}_scaler_y.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return

    model = tf.keras.models.load_model(model_path)
    scaler_y = joblib.load(scaler_path) # Load scaler giá Close

    # dự báo
    test_split_idx = int(len(X) * 0.65)
    
    X_test = X[test_split_idx:]
    y_test = y[test_split_idx:]
    
    # Model dự báo ra giá trị trong khoảng [0, 1]
    y_pred_scaled = model.predict(X_test)

    # đảo ngược chuẩn hoá về usd
    # Dùng scaler_y đã load để đổi ngược
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    y_pred_actual = scaler_y.inverse_transform(y_pred_scaled)

    # tính toán chỉ số
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mape = mean_absolute_percentage_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)

    print(f"RMSE (Sai số trung bình): ${rmse:.2f}")
    print(f"MAPE (Sai số %):          {mape*100:.2f}%")
    print(f"R2 Score (Độ khớp):       {r2:.4f}")

    # 7. VẼ BIỂU ĐỒ
    plot_results(ticker, y_test_actual, y_pred_actual, r2)

def plot_results(ticker, y_true, y_pred, r2):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, color='#00ff00', label='Giá Thực tế (Actual)') 
    plt.plot(y_pred, color='#ff0000', label='Giá Dự báo (Predicted)') 
    
    plt.title(f'Dự báo giá cổ phiếu {ticker} (Multivariate CNN-LSTM) - R2: {r2:.2f}')
    plt.xlabel('Thời gian (Ngày)')
    plt.ylabel('Giá trị (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Lưu ảnh
    if not os.path.exists('reports'):
        os.makedirs('reports')
    save_path = f"reports/{ticker}_evaluation.png"
    plt.savefig(save_path)
    print(f"\n Đã lưu biểu đồ vào: {save_path}")
    print(f"Đã lưu file tại thư mục 'reports/'.")
    plt.close()

if __name__ == "__main__":
    TICKERS = ['AAPL', 'TSLA']
    
    for ticker in TICKERS:
        evaluate_model(ticker)
        
