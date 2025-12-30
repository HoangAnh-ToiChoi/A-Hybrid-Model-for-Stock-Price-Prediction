import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(csv_path, time_step):
    # 1 doc du lieu tu file csv o folder data/raw
    df = pd.read_csv(csv_path)

    # Ép giá trị cột "Close" về dạng số
    df['Close'] = pd.to_numeric(df['Close'], errors = 'coerce')
    # Xóa các cột có giá trị rỗng
    df = df.dropna()

    #2 lay cot 'Close' de tien xu ly
    data = df.filter(['Close']).values

    # 3 chuan hoa du lieu ve khoang (0, 1)
    scaler = MinMaxScaler(feature_range=(0, 1)) # CNN-LSTM hoạt động tốt nhất khi nhận giá tị từ 0-1
    scaler_data = scaler.fit_transform(data.reshape(-1, 1)) # sắp xếp dữ liệu thành 1 cột dọc

    # 4 chia du lieu thanh 65 va 35 de train 
    strain_size = int(np.ceil(len(scaler_data) * 0.65)) #np.ceil là làm tròn lên thành số nguyên
    train_data = scaler_data[0:strain_size, :]
# Chạy từ đầu đến 65% tệp dữ liệu 

    # 5 tao dataset de train
    x_train, y_train = create_dataset(train_data, time_step)

    #6 tao dataset de test
    test_data = scaler_data[strain_size - time_step:, :] 
    x_test, y_test = create_dataset(test_data, time_step)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    return x_train, y_train, x_test, y_test, scaler 

def create_dataset(data, time_step):
    data_x, data_y = [], []
    for i in range(len(data) - time_step - 1): # chạy đến 889
        a = data[i:(i + time_step), 0] # 889 chạy đến 998 này là chỉ số index
        data_x.append(a) # thêm vào a
        data_y.append(data[i + time_step, 0]) # lưu kết quả từ 998 
    return np.array(data_x), np.array(data_y)