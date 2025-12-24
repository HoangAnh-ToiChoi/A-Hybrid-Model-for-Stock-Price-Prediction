import numpy as np
import os
import pandas as pd
from model import build_cnn_lstm_model
from config import TIME_STEP, EPOCH, BATCH_SIZE, LR, DATA_PATH, MODELS_DIR, MODEL_PATH, SCALER_PATH
from preprocess import load_data # hàm tải dữ liệu từ file csv
import glob 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import joblib

def train_model(csv_file):
    # Lấy dữ liệu từ file csv
    stock_symbol = os.path.basename(csv_file).replace('.csv', '')
    
    print(f"\n{'='*40}")
    print(f"Đang tải dữ liệu từ mã cổ phiếu: {stock_symbol}")
    print(f"{'='*40}")

    try:
        # Load dữ liệu (Đảm bảo hàm load_data trong preprocess.py trả về cả scaler)
        x_train, y_train, x_test, y_test, scaler = load_data(csv_file, TIME_STEP)
        print(f"Dữ liệu {stock_symbol} đã được tải lên và đã sẵn sàng")
        print(f"   Shape train: {x_train.shape}")
        print(f"   Shape test: {x_test.shape}")
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu. Lỗi đọc file {csv_file}", e)
        return

    print("\nBẮT ĐẦU KHỞI TẠO MODEL HYBRID")
    model = build_cnn_lstm_model(time_step=TIME_STEP, features=1, learning_rate=LR)
    
    # --- CẤU HÌNH ĐƯỜNG DẪN LƯU ---
    model_save_path = os.path.join(MODELS_DIR, f"{stock_symbol}_best_model.keras")
    scaler_save_path = os.path.join(MODELS_DIR, f"{stock_symbol}_scaler.pkl")
    
    print(f"\n🏃 QUÁ TRÌNH TRAIN DỮ LIỆU {stock_symbol} BẮT DẦU: ")

    checkpoint = [
        # SỬA LỖI 1: Bỏ f"" đi, dùng trực tiếp biến model_save_path
        ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1),
        
        # Giảm tốc độ học nếu loss đi ngang
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        
        # Dừng sớm nếu không khá hơn sau 10 epoch
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        x_train, y_train, 
        # SỬA LỖI 2: Dùng đúng tập test đã chuẩn bị, không split thêm nữa
        validation_data=(x_test, y_test), 
        epochs=EPOCH, 
        batch_size=BATCH_SIZE, 
        callbacks=checkpoint,
        verbose=1
    )

    # Lưu Scaler
    joblib.dump(scaler, scaler_save_path)
    print(f"Lưu Scaler tại: {scaler_save_path}")
    print(f"Model tốt nhất đã được tự động lưu tại: {model_save_path}")

def main():
    # Đảm bảo thư mục models tồn tại
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    csv_files = glob.glob(os.path.join(DATA_PATH, '*.csv'))

    if not csv_files:
        print(f"Không tìm thấy file CSV trong thư mục {DATA_PATH}")
        return
    
    print(f"Tìm thấy {len(csv_files)} file dữ liệu: {[os.path.basename(f) for f in csv_files]}")

    for csv_file in csv_files:
        train_model(csv_file)
        
    print("\n🎉 Đã hoàn thành training cho tất cả các mã cổ phiếu!")
    print("hehehe 🚀")

if __name__ == '__main__':
    main()