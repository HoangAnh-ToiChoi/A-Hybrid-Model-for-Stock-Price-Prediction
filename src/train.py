import numpy as np
import os
import pandas as pd
from model import build_cnn_lstm_model
from config import TIME_STEP, EPOCH, BATCH_SIZE, LR, DATA_PATH
from preprocess import load_data # hàm tải dữ liệu từ file csv
import glob 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def train_model(csv_file):
    # Lấy dữ liệu từ file csv
    stock_symbol = os.path.basename(csv_file).replace('.csv', '')
    
    print(f"\n{'='*40}")
    print(f"Đang tải dữ liệu từ mã cổ phiếu: {stock_symbol}\n")
    print(f"\n{'='*40}")

    try:
        x_train, y_train, x_test, y_test, scaler = load_data(csv_file, TIME_STEP)
        print(f"Dữ liệu đã {stock_symbol} được tải lên và đã sẵn sàng 💪 \n")
        print(f"Shape train: {x_train.shape}\n")
        print(f"Shape test: {x_test.shape}\n")
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu. Lỗi đọc file {csv_file}", e)
        return
    print("\n BẮT ĐẦU KHỞI TẠO MODEL HYBRID CNN-LSTM \n")
    model = build_cnn_lstm_model(time_step = TIME_STEP, features = 1, learning_rate = LR)
    model.summary()

    print (f"\n QUÁ TRÌNH TRAIN DỮ LIỆU {stock_symbol} BẮT DẦU: ")

    checkpoint = [
            # Lưu model tốt nhất
    ModelCheckpoint(f"experiments/{stock_symbol}.keras", monitor='val_loss', save_best_only=True, verbose=1),
    
    # KỸ THUẬT MỚI: Giảm Learning Rate khi loss đi ngang
    # Nếu val_loss không giảm sau 3 epoch -> chia đôi tốc độ học
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    
    # Dừng sớm nếu không khá hơn sau 10 epoch
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        x_train, y_train, # lấy dữ liệu và kết quả ra học 
        validation_data = (x_test, y_test), # dùng dữ liệu test và kết quả test để kiểm tra sau khi đã học hết 1000 câu hỏi
        epochs = EPOCH, # lặp lại 32 lần
        batch_size = BATCH_SIZE, # 1000 câu hỏi thì mỗi lần học chỉ 32 câu đến khi hết 1000 câu thì quay lại dòng validation_data
        callbacks = checkpoint
    )

    save_dir = 'experiments'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{stock_symbol}.keras")
    model.save(save_path)
    print(f"đã lưu model tại: {save_path}")


def main():
    csv_files = glob.glob(os.path.join(DATA_PATH, '*.csv'))

    if not csv_files:
        print(f"Không tìm thấy file CSV trong thư mục {DATA_PATH}")
        return
    
    print(f"🔍 Tìm thấy {len(csv_files)} file dữ liệu: {[os.path.basename(f) for f in csv_files]}")

    for csv_file in csv_files:
        train_model(csv_file)
    print("\n🎉 Đã hoàn thành training cho tất cả các mã cổ phiếu!")
    print("hehehe 🚀")

if __name__ == '__main__':
    main()