import numpy as np
import os
from model import build_cnn_lstm_model

def get_mock_data(time_step = 60):
    print("Test dữ liệu!")

    x_train = np.random.rand(1000, time_step, 1)
    y_train = np.random.rand(1000, 1)

    x_test = np.random.rand(00, time_step, 1)
    y_test = np.random.rand(00, 1)

    return x_train, y_train, x_test, y_test
def main():
    TIME_STEP = 60
    EPOCH = 10
    BATCH_SIZE = 32
    LR = 0.001

    x_train, y_train, x_test, y_test = get_mock_data(time_step = TIME_STEP)
    model = build_cnn_lstm_model(TIME_STEP)
    print("Khoi tao model")
    model = build_cnn_lstm_model(TIME_STEP, LR)
    model.summary()

    print ("Bat dau train: ")
    history = model.fit(
        x_train, y_train,
        validation_data = (x_test, y_test),
        epochs = EPOCH,
        batch_size = BATCH,
        verbose = 1
    )

    save_dir = 'helloo'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model.keras')
    model.save(save_path)
    print(f"đã lưu model tại: {save_path}")
if __name__ == '__main__':
    main()