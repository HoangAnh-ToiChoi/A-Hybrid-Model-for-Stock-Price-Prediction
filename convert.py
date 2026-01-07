import argparse
import os
import glob
import tensorflow as tf
import numpy as np

def convert(h5_path: str, out_path: str, quantize: str = "none"):
    if not os.path.exists(h5_path):
        print(f"[ERROR] Model file not found: {h5_path}")
        return

    print(f"Loading Keras model from: {h5_path}")
    # Load model nhưng không compile để tránh lỗi optimizer
    model = tf.keras.models.load_model(h5_path, compile=False)
    
    # Lấy thông tin đầu vào
    input_shape = model.input_shape
    print(f"Original Input shape: {input_shape}")


    # Thay vì để batch_size là None, ta ép nó về 1 (vì khi chạy dự báo ta chỉ chạy 1 mẫu)
    # Điều này giúp TFLite tối ưu hóa tốt hơn và tránh lỗi dynamic shape
    new_shape = (1, input_shape[1], input_shape[2])
    print(f"Freezing model with static shape: {new_shape}")

    run_model = tf.function(lambda x: model(x))
    
    # Tạo hàm cụ thể (Concrete Function)
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(new_shape, model.inputs[0].dtype)
    )

    # Chuyển đổi từ Concrete Function (Thay vì from_keras_model)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # cấu hình lstm
    if quantize == "float16":
        print("Applying float16 quantization")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    # Bật hỗ trợ các phép toán nâng cao (Flex Delegate)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Phép toán chuẩn
        tf.lite.OpsSet.SELECT_TF_OPS    # Phép toán TF (cho LSTM/TensorList)
    ]
    
    # Quan trọng: Tắt hạ cấp TensorList để tránh lỗi FlexTensorListReserve
    converter._experimental_lower_tensor_list_ops = False

    try:
        print("Starting conversion (This might take a few seconds)...")
        tflite_model = converter.convert()
        
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Lưu file
        with open(out_path, "wb") as f:
            f.write(tflite_model)
            
        print(f"SUCCESS: Saved TFLite model to: {out_path}")
        
    except Exception as e:
        print(f"\n❌ CONVERSION FAILED for {h5_path}")
        print(f"Error details: {e}")

def batch_convert_all(quantize: str = "none"):
    h5_files = glob.glob("models/*_final_model.h5")
    if not h5_files:
        print("No *_final_model.h5 files found in models/")
        return
    for h5_path in h5_files:
        # Lấy tên mã (VD: AAPL từ AAPL_final_model.h5)
        ticker = os.path.basename(h5_path).split("_")[0]
        out_path = f"models/{ticker}.tflite" # Đổi tên cho gọn thành AAPL.tflite
        
        print(f"\n--- Processing: {ticker} ---")
        convert(h5_path, out_path, quantize=quantize)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, default=None, help="Ticker name")
    p.add_argument("--h5", type=str, default=None, help="Path to .h5 file")
    p.add_argument("--out", type=str, default=None, help="Output path")
    p.add_argument("--quantize", type=str, choices=["none", "float16"], default="none")
    p.add_argument("--all", action="store_true", help="Convert all models")
    args = p.parse_args()

    # Tắt GPU khi convert để tránh lỗi bộ nhớ
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if args.all:
        batch_convert_all(quantize=args.quantize)
    else:
        if args.h5 is None and args.ticker is None:
            # Mặc định convert hết nếu không tham số
            batch_convert_all(quantize=args.quantize)
        else:
            if args.h5:
                h5 = args.h5
                # Tự đoán output nếu không nhập
                out = args.out if args.out else h5.replace(".h5", ".tflite")
            else:
                h5 = f"models/{args.ticker}_final_model.h5"
                out = f"models/{args.ticker}.tflite"
            
            convert(h5, out, quantize=args.quantize)