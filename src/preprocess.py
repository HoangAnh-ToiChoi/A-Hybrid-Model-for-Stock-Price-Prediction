import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, time_steps=60):
    """
    Hàm cắt dữ liệu thành các cửa sổ trượt (Sliding Window).
    Input: Mảng 2D [Giá, Giá, Giá...]
    Output: 
        - X: Mảng 3D (Samples, TimeSteps, 1) -> Dữ liệu quá khứ
        - y: Mảng 1D (Samples,) -> Giá ngày tiếp theo (Target)
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        # Lấy 60 ngày quá khứ làm Input
        X.append(data[i:(i + time_steps), 0])
        # Lấy ngày thứ 61 làm nhãn (Target) để dự đoán
        y.append(data[i + time_steps, 0])
        
    return np.array(X), np.array(y)

def process_and_save(ticker, raw_folder='data/raw', processed_folder='data/processed', time_steps=60):
    print(f" Đang xử lý mã: {ticker}...")
    
    # 1. Kiểm tra file nguồn
    file_path = os.path.join(raw_folder, f"{ticker}.csv")
    if not os.path.exists(file_path):
        print(f"Không tìm thấy file {file_path}")
        return

    # 2. Đọc dữ liệu
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Chỉ lấy cột Close (Giá đóng cửa)
    if 'Close' not in df.columns:
        print(f"File {ticker}.csv không có cột 'Close'")
        return
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna()
        
    data = df[['Close']].values # Chuyển thành numpy array
    
    # 3. Chuẩn hóa dữ liệu về khoảng [0, 1]
    # Mô hình LSTM học tốt nhất khi dữ liệu nhỏ
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # 4. Tạo Sliding Window (Cắt dữ liệu)
    X, y = create_sequences(data_scaled, time_steps)
    
    # Reshape X từ (Samples, 60) -> (Samples, 60, 1) để khớp với Input của CNN-LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    print(f"Kích thước gốc: {data.shape}")
    print(f"Kích thước sau khi cắt (X): {X.shape}")
    print(f"Kích thước nhãn (y): {y.shape}")

    # 5. Chia Train/Test (80% Train, 20% Test)
    # Cắt theo thời gian (không được random shuffle)
    train_size = int(len(X) * 0.8)
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 6. Lưu file
    save_path = os.path.join(processed_folder, ticker)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    np.save(f"{save_path}/X_train.npy", X_train)
    np.save(f"{save_path}/y_train.npy", y_train)
    np.save(f"{save_path}/X_test.npy", X_test)
    np.save(f"{save_path}/y_test.npy", y_test)
    
    joblib.dump(scaler, f"{save_path}/scaler.pkl")
    
    print(f"Đã lưu vào: {save_path}")
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape:  {X_test.shape}")

if __name__ == "__main__":
    MY_TICKERS = ['AAPL', 'TSLA']
    TIME_STEPS = 60 
    
    
    # Tạo thư mục processed nếu chưa có
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')

    for ticker in MY_TICKERS:
        process_and_save(ticker, time_steps=TIME_STEPS)
    