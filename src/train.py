import numpy as np
import os
import pandas as pd
from model import build_cnn_lstm_model
from config import TIME_STEP, EPOCH, BATCH_SIZE, LR, DATA_PATH
from preprocess import load_data 
import glob 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def train_model(csv_file):
    # Lấy tên mã chứng khoán từ tên file (VD: data/raw/AAPL.csv -> AAPL)
    stock_symbol = os.path.basename(csv_file).replace('.csv', '')
    
    print(f"\n{'='*40}")
    print(f"Đang bắt đầu huấn luyện cho mã: {stock_symbol}")
    print(f"{'='*40}\n")

    try:
        # Load dữ liệu (Đảm bảo load_data trả về đúng shape theo TIME_STEP=100)
        x_train, y_train, x_test, y_test, scaler = load_data(csv_file, TIME_STEP)
        print(f"Đã tải dữ liệu thành công!")
        print(f"- Shape Train: {x_train.shape}")
        print(f"- Shape Test:  {x_test.shape}")
    except Exception as e:
        print(f"Lỗi tải dữ liệu file {csv_file}: {e}")
        return

    # Xây dựng Model
    model = build_cnn_lstm_model(time_step=TIME_STEP, features=1, learning_rate=LR)
    # model.summary() # Có thể ẩn đi cho đỡ rối màn hình

    # Tạo thư mục lưu model nếu chưa có
    save_dir = 'models' # Nên lưu vào folder models thay vì experiments cho gọn
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    # Đường dẫn lưu file model chuẩn (.h5 để Role 3 dễ dùng)
    model_path = os.path.join(save_dir, f"{stock_symbol}_model.h5")

    # Cấu hình Callbacks
    checkpoint = [
        # Chỉ lưu model tốt nhất vào file .h5
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1),
        
        # Giảm tốc độ học nếu loss đi ngang (Fine-tuning)
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        
        # Dừng sớm để tiết kiệm thời gian
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]

    print(f"Bắt đầu training {EPOCH} epochs...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test), 
        epochs=EPOCH,
        batch_size=BATCH_SIZE,
        callbacks=checkpoint,
        verbose=1
    )

    print(f"sHoàn thành! Model đã lưu tại: {model_path}")

def main():
    # Tìm tất cả file csv trong folder data
    csv_files = glob.glob(os.path.join(DATA_PATH, '*.csv'))

    if not csv_files:
        print(f"Không tìm thấy file CSV nào trong thư mục {DATA_PATH}")
        return
    
    print(f"Tìm thấy {len(csv_files)} file dữ liệu: {[os.path.basename(f) for f in csv_files]}")

    for csv_file in csv_files:
        train_model(csv_file)
    
if __name__ == '__main__':
    main()