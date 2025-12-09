print("TEST: Python đang chạy...")
import tensorflow as tf
import yfinance as yf
import streamlit as st

print("-" * 30)
print(f"TensorFlow Version: {tf.__version__}")
print("GPU Available: ", len(tf.config.list_physical_devices('GPU')) > 0)
print("Yfinance imported successfully")
print("Streamlit imported successfully")
print("-" * 30)
