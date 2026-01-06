import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Input, Flatten
from tensorflow.keras.optimizers import Adam

def build_cnn_lstm_model(time_step, n_features, learning_rate=0.001):
    #Lớp 1;input layer
    model = Sequential()
    model.add(Input(shape=(time_step, n_features)))

    # cnn
    # tăng filters lên 64 xử lý 6 cột dữ liệu
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # lstm
    # return_sequences=True để truyền chuỗi sang lớp LSTM tiếp theo
    model.add(LSTM(128, return_sequences=True)) 
    model.add(Dropout(0.3)) # tăng dropout để chống overlifting

    # return_sequences=False để tổng hợp kết quả cuối cùng
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))

    # outputlayer
    model.add(Dense(32, activation='relu')) # Lớp trung gian
    model.add(Dense(1)) # Output: 1 giá trị (Giá Close dự báo)

    #compile
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model