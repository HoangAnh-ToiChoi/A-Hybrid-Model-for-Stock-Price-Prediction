import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib  # Dùng để load Scaler
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
# Import các biến cấu hình đường dẫn
from config import DATA_PATH, MODELS_DIR, TIME_STEP

# Cấu hình mã cần test (Bạn có thể đổi thành TSLA)
STOCK_SYMBOL = "AAPL" 

# Đường dẫn chuẩn theo Phương án 2
MODEL_FILE = os.path.join(MODELS_DIR, f"{STOCK_SYMBOL}_best_model.keras")
SCALER_FILE = os.path.join(MODELS_DIR, f"{STOCK_SYMBOL}_scaler.pkl")
CSV_FILE = os.path.join(DATA_PATH, f"{STOCK_SYMBOL}.csv")

def main():
    print(f"\n{'='*40}")
    print(f"DỰ BÁO CỔ PHIẾU: {STOCK_SYMBOL}")
    print(f"{'='*40}")

    # 1. KIỂM TRA FILE
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        print(f"Không tìm thấy model hoặc scaler tại {MODELS_DIR}")
        return

    # 2. LOAD MODEL & SCALER (QUAN TRỌNG: Load chứ không Fit lại)
    
    model = load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("Load thành công!")

    # 3. LOAD & XỬ LÝ DỮ LIỆU
    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip()
    
    if 'Date' not in df.columns:
        if 'date' in df.columns:
            df.rename(columns={'date': 'Date'}, inplace=True) 
        elif df.index.name == 'Date':
             df.reset_index(inplace=True) 
        else:
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    # In ra để kiểm tra (Debug)
    print(f"Các cột trong file CSV: {df.columns.tolist()}")
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce') # Ép kiểu ngày tháng cho đẹp
    df = df.dropna(subset=['Date'])
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    
    # Tính SMA để khuyến nghị
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()

    # Lấy dữ liệu giá để đưa vào model
    data = df.filter(['Close']).values
    # Scale dữ liệu bằng scaler đã load (KHÔNG fit lại)
    scaled_data = scaler.transform(data)

    # PHẦN 1: DỰ ĐOÁN TƯƠNG LAI 5 NGÀY (MULTI-STEP)

    # Số ngày muốn dự đoán
    FUTURE_DAYS = 5
    
    # Lấy dữ liệu 60 ngày cuối cùng (đã scale)
    current_batch = scaled_data[-TIME_STEP:].reshape(1, TIME_STEP, 1)
    
    future_predictions = [] # Chứa kết quả dự đoán (đã scale)
    
    print(f"\n Đang chạy dự báo cuốn chiếu cho {FUTURE_DAYS} ngày tới...")

    for i in range(FUTURE_DAYS):
        # 1. Dự đoán ngày tiếp theo
        next_pred = model.predict(current_batch, verbose=0)
        
        # 2. Lưu kết quả
        future_predictions.append(next_pred[0, 0])
        
        # 3. Cập nhật cửa sổ trượt (Bỏ ngày đầu, thêm ngày vừa dự đoán vào cuối)
        # next_pred đang là (1,1), cần reshape để nối vào
        next_step = next_pred.reshape(1, 1, 1)
        current_batch = np.append(current_batch[:, 1:, :], next_step, axis=1)

    # Đảo ngược scale để ra giá USD thật
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_prices = scaler.inverse_transform(future_predictions)
    
    # Hiển thị kết quả ra màn hình
    current_price = data[-1][0]
    last_date_obj = df.iloc[-1]['Date']
    
    print(f"\nDữ liệu cập nhật đến: {last_date_obj.strftime('%d-%m-%Y')}")
    print(f"Giá đóng cửa hiện tại: {current_price:.2f} USD")
    print("-" * 30)
    print("DỰ BÁO 5 NGÀY TIẾP THEO:")
    
    next_dates = []
    for i in range(FUTURE_DAYS):
        # Cộng thêm i+1 ngày
        next_date = last_date_obj + pd.Timedelta(days=i+1)
        next_dates.append(next_date)
        price = future_prices[i][0]
        print(f"   - Ngày {next_date.strftime('%d-%m-%Y')}: {price:.2f} USD")
    print("-" * 30)

    # Logic khuyến nghị SMA (Giữ nguyên)
    sma_10 = df.iloc[-1]['SMA_10']
    sma_60 = df.iloc[-1]['SMA_60']
    trend = "TĂNG" if sma_10 > sma_60 else "GIẢM"
    action = "MUA (Golden Cross)" if sma_10 > sma_60 else "BÁN (Death Cross)"
    
    print(f"\nPHÂN TÍCH KỸ THUẬT:")
    print(f"Xu hướng: {trend}")
    print(f"Khuyến nghị: {action}")

    # PHẦN 2: KIỂM TRA ĐỘ CHÍNH XÁC & VẼ BIỂU ĐỒ
    
    print("\n Vẽ biểu đồ kiểm chứng...")
    
    # ... (Đoạn code chuẩn bị dữ liệu test cũ giữ nguyên) ...
    test_len = 100
    if len(scaled_data) < test_len + TIME_STEP:
        test_len = len(scaled_data) - TIME_STEP

    x_test_plot = []
    for i in range(len(scaled_data) - test_len, len(scaled_data)):
        x_test_plot.append(scaled_data[i-TIME_STEP:i, 0])
    
    x_test_plot = np.array(x_test_plot)
    x_test_plot = np.reshape(x_test_plot, (x_test_plot.shape[0], x_test_plot.shape[1], 1))
    
    predictions = model.predict(x_test_plot, verbose=0)
    predictions = scaler.inverse_transform(predictions)

    valid_data = df.iloc[-test_len:].copy()
    valid_data['Predictions'] = predictions

    # --- VẼ HÌNH (VẼ 5 ĐIỂM) ---
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'].tail(150), df['Close'].tail(150), label='Lịch sử giá', color='gray', alpha=0.5)
    plt.plot(valid_data['Date'], valid_data['Close'], label='Giá Thực Tế (Test)', color='blue', linewidth=2)
    plt.plot(valid_data['Date'], valid_data['Predictions'], label='AI Dự Đoán (Backtest)', color='red', linestyle='--', linewidth=2)

    # VẼ 5 ĐIỂM DỰ BÁO TƯƠNG LAI
    # Vẽ đường nối từ giá hiện tại đến dự đoán ngày đầu tiên
    plt.plot([last_date_obj, next_dates[0]], [current_price, future_prices[0][0]], color='lime', linestyle=':', linewidth=2)
    
    # Vẽ chuỗi 5 ngày
    plt.plot(next_dates, future_prices, color='lime', marker='o', markersize=8, label='Dự báo 5 ngày tới', linewidth=2, zorder=10)

    # Hiện giá lên từng điểm
    for i in range(FUTURE_DAYS):
        plt.text(next_dates[i], future_prices[i][0], f'{future_prices[i][0]:.1f}', fontsize=9, ha='center', va='bottom')

    plt.title(f"Dự báo {STOCK_SYMBOL}: 5 Ngày Tiếp Theo", fontsize=16)
    plt.xlabel('Ngày', fontsize=12)
    plt.ylabel('Giá (USD)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    save_img = f"experiments/{STOCK_SYMBOL}_validation_chart.png"
    plt.savefig(save_img)
    print(f"Lưu biểu đồ kiểm chứng tại: {save_img}")

    
    # plt.show() 

if __name__ == "__main__":
    main()