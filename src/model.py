import tensorflow as tf
from tensorflow.keras.models import Sequential # mo hinh tuyen tinh, cac lop xep chong len nhau
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D # cac lop bulding block


def build_cnn_lstm_model(time_step = 60, features = 1, leaning_rate = 0.001):
    model = Sequential() # tạo ra 1 model xếp lớp
    # 1. Lớp CNN(trích xuất đặc trưng từ biểu đồ giá)
    model.add(Input(shape=(time_step, features))) # 60 dòng dữ liệu thì mỗi dòng có 1 giá trị 
    model.add(Conv1D(filters=64, kernel_size = 2, activation = 'relu')) # chia 64 layer mỗi bước quét 2 ngày, giúp mô hình tuyến tính 
    model.add(MaxPooling1D(pool_size=2))  # gom 2 giá trị thành 1 giá trị lớn nhất 

    # 2. Lớp LSTM(học quy luật thời gian) 
    # LSTM thứ 1
    model.add(LSTM(100, return_sequences = True))
    model.add(Dropout(0.2))
    # LSTM thứ 2
    model.add(LSTM(50, return_sequences = False))
    model.add(Dropout(0.2))

    # 3. Out put
    model.add(Dense(25, activation = 'relu'))
    model.add (Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss = 'mse', optimizer = optimizer)
    
    return model