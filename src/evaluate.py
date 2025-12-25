import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import os

# --- TẮT GPU ĐỂ TRÁNH TREO MÁY TRÊN MAC ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass
# ------------------------------------------

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from preprocess import load_data
from config import TIME_STEP, DATA_PATH

def evaluate_model(ticker):
    print(f" ĐANG ĐÁNH GIÁ MÃ: {ticker}")


    csv_file = os.path.join(DATA_PATH, f"{ticker}.csv")
    model_path = f"models/{ticker}_model.h5"
    scaler_path = f"data/processed/{ticker}/scaler.pkl"

    if not os.path.exists(model_path):
        print(f" Không tìm thấy model: {model_path}")
        return

    try:
        _, _, X_test, y_test, scaler = load_data(csv_file, TIME_STEP)
        model = tf.keras.models.load_model(model_path)
        
        print(" Đang chạy dự đoán (Direct Call Mode)...")
        # Gọi model trực tiếp để tránh treo máy
        y_pred_scaled = model(X_test, training=False).numpy()
        
        print("4. Đang tính toán chỉ số...")
        
       
        # Ép kiểu dữ liệu về dạng cột dọc (2D array) để Scaler hiểu
        y_test = np.array(y_test).reshape(-1, 1)
        y_pred_scaled = np.array(y_pred_scaled).reshape(-1, 1)

        y_test_actual = scaler.inverse_transform(y_test)
        y_pred_actual = scaler.inverse_transform(y_pred_scaled)

        # Tính toán các chỉ số
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        mape = mean_absolute_percentage_error(y_test_actual, y_pred_actual)
        r2 = r2_score(y_test_actual, y_pred_actual)

        print(f"\n   KẾT QUẢ ĐÁNH GIÁ CHO {ticker}:")
        print(f"   - RMSE (Sai số trung bình): ${rmse:.2f}")
        print(f"   - MAPE (Sai số %):          {mape*100:.2f}%")
        print(f"   - R2 Score (Độ chính xác):  {r2:.4f}")

        plot_results(ticker, y_test_actual, y_pred_actual)
        
    except Exception as e:
        print(f" Lỗi chi tiết: {e}")
        import traceback
        traceback.print_exc()

def plot_results(ticker, y_true, y_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, color='blue', label='Thực tế')
    plt.plot(y_pred, color='red', label='Dự đoán', alpha=0.7)
    plt.title(f'So sánh giá {ticker} (Test Set)')
    plt.legend()
    plt.grid(True)
    
    if not os.path.exists('reports'):
        os.makedirs('reports')
    plt.savefig(f"reports/{ticker}_evaluation.png")
    plt.close()

if __name__ == "__main__":
    for t in ["AAPL", "TSLA"]:
        evaluate_model(t)