import os
import joblib
import glob
# Import hàm load_data từ file preprocess cũ của bạn
from preprocess import load_data
# Import cấu hình
from config import DATA_PATH, TIME_STEP

def main():
    # Tìm tất cả file csv gốc
    csv_files = glob.glob(os.path.join(DATA_PATH, '*.csv'))
    
    if not csv_files:
        print("Không tìm thấy file CSV nào.")
        return

    print(f"Tìm thấy {len(csv_files)} mã cổ phiếu. Đang tạo scaler...")

    for file in csv_files:
        # Lấy tên mã (VD: AAPL)
        ticker = os.path.basename(file).replace('.csv', '')
        
        try:
            # Gọi hàm load_data để nó tự fit và trả về scaler
            # (Chúng ta không cần x_train, y_train nên dùng dấu _)
            _, _, _, _, scaler = load_data(file, TIME_STEP)
            
            # Tạo đường dẫn lưu: data/processed/AAPL/
            save_folder = f'data/processed/{ticker}'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            
            # Lưu file scaler.pkl
            save_path = os.path.join(save_folder, 'scaler.pkl')
            joblib.dump(scaler, save_path)
            
            print(f"Đã lưu: {save_path}")
            
        except Exception as e:
            print(f"Lỗi xử lý {ticker}: {e}")

if __name__ == '__main__':
    main()