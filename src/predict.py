import numpy as np 
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler

MODEL_PATH = 'helloo/model.keras'

def main():
    if not os.path.exists(MODEL_PATH):
        print("Chua co file model kia cu!")
        return
    
    print("Dang nap Model ...")
    model = tf.keras.models.load_model(MODEL_PATH)

    dummy_input = np.random.rand(1, 60, 1)

    pre_scaled_price = model.predict(dummy_input)
    val = pre_scaled_price[0][0]

    print("-" * 30)
    print(f"🔮 Giá trị Model trả về (Scaled): {val:.4f}") 
    print("(Đây là con số máy hiểu, nằm trong khoảng 0-1)")

    min_price = 100
    max_price = 200

    real_price = val * (max_price - min_price) + min_price
    print(f"💰 Giá tiền thật quy đổi: ${real_price:.2f}")
    print("-" * 30)

if __name__ == '__main__':
    main()