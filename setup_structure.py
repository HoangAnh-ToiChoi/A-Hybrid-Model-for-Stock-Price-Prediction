import os

# Danh sách thư mục cần tạo
folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "app",
    "models",
    "experiments",
    "tests"
]

# Danh sách file rỗng cần tạo
files = [
    "src/__init__.py",
    "src/data_ingest.py",
    "src/preprocess.py",
    "src/model.py",
    "src/train.py",
    "src/evaluate.py",
    "app/app.py",
    "experiments/exp1.yaml",
    "README.md"
]

# Tạo thư mục
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✅ Đã tạo thư mục: {folder}")

# Tạo file
for file in files:
    if not os.path.exists(file):
        with open(file, 'w') as f:
            pass # Tạo file rỗng
        print(f"✅ Đã tạo file: {file}")
    else:
        print(f"⚠️ File đã tồn tại: {file}")

print("\n🎉 Cấu trúc dự án đã sẵn sàng!")