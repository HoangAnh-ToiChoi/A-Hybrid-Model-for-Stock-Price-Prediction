import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
from config import TIME_STEP

STOCK_SYMBOL = "TSLA"
MODEL_PATH = f"models/{STOCK_SYMBOL}_model.h5"
CSV_PATH = f"data/raw/{STOCK_SYMBOL}.csv"

# 1. Hàm xử lý dữ liệu
def get_scaler_and_data(csv_path):
    # 1. Load dữ liệu
    df = pd.read_csv(csv_path)
    
    # 2. Ép kiểu cột Close về số (Quan trọng!)
    # errors='coerce' sẽ biến chữ "AAPL" thành NaN (Not a Number)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    # 3. Xóa các dòng bị lỗi 
    df = df.dropna(subset=['Close'])
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    # 4. Lấy dữ liệu sạch
    data = df.filter(['Close']).values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data) 
    
    return scaler, data, df

def main():
    print(f"Mã cổ phiếu: {STOCK_SYMBOL}")
    # 1 kiểm tra model
    if not os.path.exists(MODEL_PATH):
        print(f"Chưa có model cho {STOCK_SYMBOL}. Vui lòng chạy train.py trước.")
        return
    print("Dữ liệu và mô hình đang được tải")
    model = load_model(MODEL_PATH)
    scaler, data, df_o = get_scaler_and_data(CSV_PATH)

    # 2. Lấy dữ liệu gần nhất
    # lấy 60 ngày để dự đoán ngày thứ thứ 61
    last_60days = data [-TIME_STEP:]
    input_scaled = scaler.transform(last_60days)

    x_test = np.array([input_scaled])
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # 3. Dự đoán giá
    pred_scaled = model.predict(x_test, verbose=0)
    pred_price = scaler.inverse_transform(pred_scaled)[0][0]

    # 4. So sánh với giá hiện tại 
    current_price = data[-1][0]
    last_date = df_o.iloc[-1]['Date'] if 'Date' in df_o.columns else "Phiên gần nhất"

    print(f"Dữ liệu cập nhật đến ngày: {last_date}")
    print(f"Giá đóng cửa hiện tại:{current_price:.2f} USD")
    print(f"Dự đoán phiên tiếp theo: {pred_price:.2f} USD")
    
   # Lấy giá trị SMA ngày cuối cùng
    last_sma_10 = df_o.iloc[-1]['SMA_10']
    last_sma_60 = df_o.iloc[-1]['SMA_60']

    print(f"CHỈ SỐ KỸ THUẬT (SMA):")
    print(f"SMA 10 (Ngắn hạn): {last_sma_10:.2f}")
    print(f"SMA 60 (Dài hạn):  {last_sma_60:.2f}")
    print("-" * 40)

    if pd.isna(last_sma_60):
        print("Chưa đủ dữ liệu 60 ngày để tính SMA.")
    elif last_sma_10 > last_sma_60:
        # SMA 10 > SMA 60 => MUA 
        print("MUA")
    else:
        # SMA 60 > SMA 10 => BÁN 
        print("Bán")

    # --- VẼ HÌNH ĐỂ BÁO CÁO (Output Chart) ---
    print("vẽ biểu đồ kết quả...")
    
    # Lấy 100 ngày cuối
    plot_df = df_o.tail(100).copy()
    
    plt.figure(figsize=(12, 6))
    
    # Vẽ đường giá và SMA
    plt.plot(plot_df['Close'].values, label='Giá thực tế', color='skyblue', linewidth=2)
    plt.plot(plot_df['SMA_10'].values, label='SMA 10', color='orange', linestyle='--')
    plt.plot(plot_df['SMA_60'].values, label='SMA 60', color='purple', linestyle='--')
    
    # Vẽ điểm dự đoán (QUAN TRỌNG: Dùng biến pred_price)
    # Reset index để vẽ tiếp nối vào cuối biểu đồ
    last_idx = len(plot_df) 
    plt.scatter(last_idx, pred_price, color='red', s=150, zorder=5, 
                label=f'AI Dự đoán: {pred_price:.2f}') # <--- Hiện số lên Legend
    
    # Vẽ đường nối đứt đoạn từ giá cuối đến giá dự đoán
    plt.plot([last_idx-1, last_idx], [plot_df['Close'].iloc[-1], pred_price], 
             color='red', linestyle=':', alpha=0.6)

    plt.title(f'Dự báo {STOCK_SYMBOL}')
    plt.legend()
    plt.grid(True, alpha=0.3)    
    # Lưu ảnh
    save_path = f"experiments/{STOCK_SYMBOL}_prediction.png"
    plt.savefig(save_path)
    print(f"Lưu ảnh tại: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()