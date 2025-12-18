import yfinance as yf
import pandas as pd
import os

def download_stock_data(ticker, start_date, end_date, save_folder):
    """
    Hàm tải dữ liệu và lưu vào thư mục chỉ định
    """
    print(f"\n Đang xử lý mã: {ticker}...")
    
    #Call data từ Yahoo Finance
    try: 
        df = yf.download(ticker, start=start_date, end=end_date)
    except Exception as e:
        print(f" Lỗi kết nối khi tải {ticker}: {e}")
        return None
    
    # Kiểm tra dữ liệu
    if len(df) == 0:
        print(f" Không tìm thấy dữ liệu cho {ticker}! Vui lòng kiểm tra lại mã.")
        return None
    
    # Chỉ giữ lại cột Close

    df = df[['Close']]
    
    # 4. Tạo đường dẫn lưu file (
    os.makedirs(save_folder, exist_ok=True)
    file_path = os.path.join(save_folder, f"{ticker}.csv")
    
    # 5. Lưu ra file CSV
    df.to_csv(file_path)
    print(f" Đã lưu thành công: {file_path}")
    print(f" Tổng số dòng: {len(df)}")
    return df

#  (MAIN) 
if __name__ == "__main__":
    # CẤU HÌNH: Danh sách mã cổ phiếu muốn tải
    MY_TICKERS = ['AAPL', 'TSLA'] 
    START_DATE = '2015-01-01'
    END_DATE = '2024-01-01'
    SAVE_DIR = 'data/raw' # Thư mục lưu trữ
    
    print(" BẮT ĐẦU TẢI DỮ LIỆU...")
    
    # Vòng lặp: Chạy qua từng mã trong danh sách
    for ticker in MY_TICKERS:
        download_stock_data(ticker, START_DATE, END_DATE, SAVE_DIR)
        
    print(" HOÀN THÀNH TÁC VỤ ")