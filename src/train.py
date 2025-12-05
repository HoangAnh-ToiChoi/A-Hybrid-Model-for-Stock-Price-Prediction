import numpy as np
import os
from model import build_cnn_lstm_model
from config import TIME_STEP, EPOCH, BATCH_SIZE, LR

def get_mock_data(time_step = 60):
    print("Test dữ liệu!")

    x_train = np.random.rand(1000, time_step, 1) # dùng kỹ thuật trượt. Học 1000 bộ, mỗi bộ ghi lại diễn biến trong 60 ngày
    y_train = np.random.rand(1000, 1)

    x_test = np.random.rand(100, time_step, 1)
    y_test = np.random.rand(100, 1)

    return x_train, y_train, x_test, y_test
def main():


    x_train, y_train, x_test, y_test = get_mock_data(time_step = TIME_STEP) 
    
    print("Khoi tao model")
    model = build_cnn_lstm_model(time_step = TIME_STEP, features = 1, learning_rate = LR)
    model.summary()

    print ("Bat dau train: ")
    history = model.fit(
        x_train, y_train, # lấy dữ liệu và kết quả ra học 
        validation_data = (x_test, y_test), # dùng dữ liệu test và kết quả test để kiểm tra sau khi đã học hết 1000 câu hỏi
        epochs = EPOCH, # lặp lại 32 lần
        batch_size = BATCH_SIZE, # 1000 câu hỏi thì mỗi lần học chỉ 32 câu đến khi hết 1000 câu thì quay lại dòng validation_data
        verbose = 1 # sai số càng thấp thì càng khôn
    )

    save_dir = 'helloo'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model.keras')
    model.save(save_path)
    print(f"đã lưu model tại: {save_path}")

if __name__ == '__main__':
    main()