import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.config import FEATURE_COLUMNS # Import danh sách cột

def create_dataset_vectorized(dataset, time_step=60):
    
    # Chuyển dữ liệu thành dạng mảng numpy nếu chưa phải
    data = np.array(dataset)
    
    # Tính số lượng mẫu có thể tạo ra
    n_samples = len(data) - time_step
    
    if n_samples <= 0:
        return np.array([]), np.array([])

    # Ma thuật của Numpy: Tạo các chỉ số (indexes) để cắt dữ liệu cùng lúc
    # Tạo mảng chỉ số [0, 1, 2, ..., time_step-1]
    # Sau đó cộng với [0, 1, 2, ..., n_samples-1] để tạo cửa sổ trượt
    idx = np.arange(time_step)[None, :] + np.arange(n_samples)[:, None]
    
    # Cắt X (Input) siêu tốc
    X = data[idx]
    
    # Cắt y (Output) - Lấy giá Close (cột 0) tại thời điểm ngay sau cửa sổ
    y = data[time_step:, 0]
    
    return X, y

def prepare_multivariate_data(df, time_step):
  
    # 1. Lọc cột
    df_filtered = df[FEATURE_COLUMNS]
    data_values = df_filtered.values

    # 2. Scaling
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_X.fit_transform(data_values)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.fit(df[['Close']])

    # 3. Tạo dataset bằng hàm Vectorized mới
    X, y = create_dataset_vectorized(scaled_data, time_step)

    print(f" [Fast Process] Đã xử lý {len(X)} mẫu dữ liệu trong tíc tắc.")
    return X, y, scaler_y, scaler_X