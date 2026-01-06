

import argparse
import os
import glob
import tensorflow as tf

def convert(h5_path: str, out_path: str, quantize: str = "none"):
    if not os.path.exists(h5_path):
        print(f"[ERROR] Model file not found: {h5_path}")
        return
    print(f"Loading Keras model from: {h5_path}")
    model = tf.keras.models.load_model(h5_path, compile=False)
    print(f"Model input shape: {model.input_shape}")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize == "float16":
        print("Applying float16 quantization (reduced size, faster on CPU)")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == "none":
        print("No quantization: exporting float32 TFLite model")
    else:
        raise ValueError("Unsupported quantization type. Use 'none' or 'float16'.")
    # Enable Select TF Ops for LSTM support
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"[ERROR] Failed to convert {h5_path}: {e}")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to: {out_path}")
    print(f"Output shape: {model.output_shape}")

def batch_convert_all(quantize: str = "none"):
    h5_files = glob.glob("models/*_final_model.h5")
    if not h5_files:
        print("No *_final_model.h5 files found in models/")
        return
    for h5_path in h5_files:
        ticker = os.path.basename(h5_path).split("_")[0]
        out_path = f"models/{ticker}.tflite"
        print(f"\n--- Converting {h5_path} to {out_path} ---")
        convert(h5_path, out_path, quantize=quantize)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, default=None, help="Ticker name (uses models/{ticker}_final_model.h5)")
    p.add_argument("--h5", type=str, default=None, help="Path to .h5 model file")
    p.add_argument("--out", type=str, default=None, help="Output .tflite path")
    p.add_argument("--quantize", type=str, choices=["none", "float16"], default="none", help="Quantization type")
    p.add_argument("--all", action="store_true", help="Convert all *_final_model.h5 in models/")
    args = p.parse_args()

    if args.all:
        batch_convert_all(quantize=args.quantize)
    else:
        if args.h5 is None and args.ticker is None:
            p.error("Please provide --h5 or --ticker or --all")
        if args.h5 is not None:
            h5_path = args.h5
            ticker = os.path.basename(h5_path).split("_")[0]
        else:
            h5_path = f"models/{args.ticker}_final_model.h5"
            ticker = args.ticker
        out_path = args.out if args.out is not None else f"models/{ticker}.tflite"
        convert(h5_path, out_path, quantize=args.quantize)
