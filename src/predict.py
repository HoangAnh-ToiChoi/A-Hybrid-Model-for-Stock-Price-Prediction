import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

STOCK_SYMBOL = "TSLA"
MODEL_PATH = f"experiments/{STOCK_SYMBOL}.keras"
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

    # 4. Lấy dữ liệu sạch
    data = df.filter(['Close']).values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data) 
    
    return scaler, data, df

def main():
    print(f"\n=============================================")
    print(f"   Mã cổ phiếu: {STOCK_SYMBOL}")
    print(f"=============================================\n")

    # 1 kiểm tra model
    if not os.path.exists(MODEL_PATH):
        print(f"Chưa có model cho {STOCK_SYMBOL}. Vui lòng chạy train.py trước.")
        return
    print("Dữ liệu và mô hình đang được tải")
    model = load_model(MODEL_PATH)
    scaler, data, df_o = get_scaler_and_data(CSV_PATH)

    # 2. Lấy dữ liệu gần nhất
    # lấy 60 ngày để dự đoán ngày thứ thứ 61
    last_60days = data [-60:]
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
    print(f"Giá đóng cửa hiện tại:      {current_price:.2f} USD")
    print(f"Dự đoán phiên tiếp theo: {pred_price:.2f} USD")
    
    print("\n=============================================")
    # Logic tư vấn đơn giản
    diff = pred_price - current_price
    percent = (diff / current_price) * 100

    if percent > 1.0:
        print(f"XU HƯỚNG ĐANG TĂNG MẠNH (+{percent:.2f}%)")
        print("Khuyến nghị: Cân nhắc MUA VÀO")
    elif percent > 0:
        print(f"XU HƯỚNG ĐANG TĂNG NHẸ (+{percent:.2f}%)")
        print("💡 Khuyến nghị: Nắm giữ / Mua thăm dò")
    elif percent > -1.0:
        print(f"XU HƯỚNG ĐANG GIẢM NHẸ ({percent:.2f}%)")
        print("Khuyến nghị: Thận trọng / Quan sát")
    else:
        print(f"XU HƯỚNG ĐANG GIẢM MẠNH ({percent:.2f}%)")
        print("Khuyến nghị: Cân nhắc BÁN RA")
    print("=============================================\n")

if __name__ == "__main__":
    main()