import tensorflow as tf
from tensorflow.keras.models import Sequential # mo hinh tuyen tinh, cac lop xep chong len nhau
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, LSTM, Input # cac lop bulding block


def build_cnn_lstm_model(time_step = 100, features = 1, learning_rate = 0.001): #Khởi tạo mô hình
   model = Sequential() #tạo 1 biến model dưới dạng dạng chuỗi (là các lớp nối tiếp nhau)

   model.add(Input(shape=(time_step, features))) # định nghĩa đầu vào làn 1 ma trận có kích thước là 100 và 1 (100 là tgian, 1 là 1 cột giá trị )
   model.add(Conv1D(filters = 64, kernel_size = 2, activation = 'relu')) # thêm 1 lớp tích chập 1 chiều có cửa sổ trượt là 2, bỏ giá trị âm, 64 bộ lọc để xác định đặc trưng
   model.add(MaxPooling1D(pool_size = 2)) # lấy giá trị lớn nhất trên 1 lớp có độ rộng là 2

   model.add(LSTM(64, return_sequences = True)) # lớp lstm thứ 1 có 64 đơn vị và kết quả tra ra là dạng chuỗi
   model.add(Dropout(0.2)) # ngẫu nhiên cắt 20 % đơn vị ở trên trong quá trình train
# Học những đặc trưng ngắn hạn --> trả ra chuỗi kết quả

   model.add(LSTM(32, return_sequences = False)) # lớp lstm thứ 2 có 32 đơn vị và trả kết quả là 1 vector kết quả
   model.add(Dropout(0.2)) # ngẫu nhiên cắt 20 % đơn vị ở trên trong quá trình train
# Học từ lớp thứ nhất để suy ra xu hướng và chốt kết quả cuối  

   model.add(Dense(1)) # đầu ra là 1 cột 

   optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate) # thuật toán tối ưu hóa dựa trên trọng số learning rate
   model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer = optimizer) # cách học 

# Dữ liệu vào (Input) 
# CNN (Lọc đặc trưng quan trọng) 
# LSTM 1 (Học chuỗi, giữ lại sequence) 
# LSTM 2 (Học chuỗi, tóm tắt lại thành 1 state cuối) 
# Dense (Ra 1 con số dự đoán).

   return model