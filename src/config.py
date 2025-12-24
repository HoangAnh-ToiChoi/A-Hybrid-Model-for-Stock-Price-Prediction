import os

TIME_STEP = 100
EPOCH = 100 # số lần học là đi học lại là 100 lần 
BATCH_SIZE = 32 # 32 bài học cùng lúc 
LR = 0.0001 # tốc độ học 
DATA_PATH = 'data/raw'


# ... (Các biến cũ như EPOCH, LR giữ nguyên) ...

# CẤU HÌNH LƯU MODEL & SCALER (Thêm mới)
MODELS_DIR = "models"
# Tự động tạo thư mục models nếu chưa có
os.makedirs(MODELS_DIR, exist_ok=True)

# Tên file model chuẩn để giao cho Role 3
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras") 

# Tên file Scaler (Quan trọng để Inverse Transform)
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")