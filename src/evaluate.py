import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

# Cấu hình
STOCK_SYMBOL = "TSLA" # Hoặc AAPL
MODEL_PATH = f"experiments/{STOCK_SYMBOL}.keras"
CSV_PATH = f"data/raw/{STOCK_SYMBOL}.csv"

# Load Model & Data
model = load_model(MODEL_PATH)
df = pd.read_csv(CSV_PATH)
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close'])
data = df.filter(['Close']).values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Lấy dữ liệu test (Ví dụ 20% cuối cùng)
test_len = int(len(scaled_data) * 0.2)
test_data = scaled_data[-test_len:]

x_test, y_true = [], []
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    y_true.append(test_data[i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Dự đoán
preds = model.predict(x_test, verbose=0)
# Đảo ngược scale về giá thật
preds_real = scaler.inverse_transform(preds)
y_true_real = scaler.inverse_transform(np.array(y_true).reshape(-1, 1))

# TÍNH TOÁN ĐỘ CHÍNH XÁC
mape = mean_absolute_percentage_error(y_true_real, preds_real)
accuracy = 100 - (mape * 100)

print(f"===================================")
print(f"Sai số trung bình (MAPE): {mape*100:.2f}%")
print(f"ĐỘ CHÍNH XÁC MÔ HÌNH:    {accuracy:.2f}%")
print(f"===================================")