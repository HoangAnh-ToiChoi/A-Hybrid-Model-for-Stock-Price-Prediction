import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Stock AI Predictor", layout="wide")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.data_ingest import get_realtime_data, add_technical_indicators
from src.config import TIME_STEP, FEATURE_COLUMNS

# tflite
try:
    from tensorflow.lite.python.interpreter import Interpreter as TFLiteInterpreter
except ImportError:
    st.error("Không tìm thấy TFLite Interpreter. ")
    TFLiteInterpreter = None

def load_model_and_scaler(ticker):
    tflite_path = f"models/{ticker}.tflite"
    keras_path = f"models/{ticker}_final_model.h5"
    scaler_path = f"models/{ticker}_scaler_y.pkl"
    scaler_y = None
    if not os.path.exists(scaler_path):
        return None, None, f"Không tìm thấy scaler: {scaler_path}", None
    try:
        scaler_y = joblib.load(scaler_path)
    except Exception as e:
        return None, None, f"Lỗi load scaler: {e}", None
    # Prefer TFLite for AAPL
    if ticker == "AAPL" and os.path.exists(tflite_path):
        try:
            interpreter = TFLiteInterpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            return interpreter, scaler_y, None, "tflite"
        except Exception as e:
            return None, None, f"Lỗi load TFLite: {e}", None
    # Fallback to Keras for TSLA or if TFLite missing
    if os.path.exists(keras_path):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(keras_path, compile=False)
            return model, scaler_y, None, "keras"
        except Exception as e:
            return None, None, f"Lỗi load Keras model: {e}", None
    return None, None, f"Không tìm thấy model cho {ticker}", None

# --- INFERENCE WRAPPERS ---
def run_model_inference(model, X_input: np.ndarray, mode: str) -> np.ndarray:
    if mode == "tflite":
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        idx = input_details[0]['index']
        expected_dtype = input_details[0]['dtype']
        arr = np.ascontiguousarray(X_input, dtype=expected_dtype)
        try:
            model.resize_tensor_input(idx, arr.shape, strict=False)
            model.allocate_tensors()
        except Exception:
            pass
        model.set_tensor(idx, arr)
        model.invoke()
        out_idx = output_details[0]['index']
        result = model.get_tensor(out_idx)
        return result
    elif mode == "keras":
        return model.predict(X_input, verbose=0)
    else:
        raise ValueError("Unknown model mode")

# --- INPUT PREP ---
def get_prediction_input(df, time_step):
    df_filtered = df[FEATURE_COLUMNS]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_filtered.values)
    last_100_days = df_filtered.values[-time_step:]
    scaled_data = scaler.transform(last_100_days)
    X_input = scaled_data.reshape(1, time_step, scaled_data.shape[1])
    return X_input.astype(np.float32)

st.sidebar.header("CẤU HÌNH")
ticker = st.sidebar.selectbox("Chọn mã cổ phiếu:", ["AAPL", "TSLA"])
years = st.sidebar.slider("Dữ liệu quá khứ (Năm):", 1, 5, 2)

st.title(f" Dự báo Chứng khoán: {ticker}")

try:
    with st.spinner("Đang tải dữ liệu..."):
        df = get_realtime_data(ticker, years=years)
        df = add_technical_indicators(df)
except Exception as e:
    st.error(f"Không thể tải dữ liệu: {e}")
    st.stop()

# --- LOAD MODEL ---
model, scaler_y, model_err, model_mode = load_model_and_scaler(ticker)

# --- TABS ---
tab1, tab2, tab3 = st.tabs([" Thị trường", " Dự báo", " Báo cáo"])

with tab1:
    current_price = df['Close'].iloc[-1]
    last_date = df.index[-1].strftime('%d-%m-%Y')
    change = current_price - df['Close'].iloc[-2]
    pct_change = (change / df['Close'].iloc[-2]) * 100
    col1, col2 = st.columns(2)
    col1.metric("Ngày cập nhật", last_date)
    col1.metric("Giá đóng cửa", f"${current_price:.2f}", f"{change:.2f} ({pct_change:.2f}%)")
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'
    )])
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'))
    fig.update_layout(height=450, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Dự báo phiên tiếp theo")
    if st.button(" RUN ", type="primary", use_container_width=True):
        if model is None or scaler_y is None:
            st.error(model_err or "Không tìm thấy model hoặc scaler cho mã này")
        else:
            with st.spinner("Mô hình đang xử lý..."):
                try:    
                    X_input = get_prediction_input(df, TIME_STEP)
                    pred_scaled = run_model_inference(model, X_input, model_mode)
                    pred_arr = np.array(pred_scaled).reshape(-1, 1)
                    pred_price = scaler_y.inverse_transform(pred_arr)[0][0]
                except Exception as e:
                    st.error(f"Lỗi inference: {e}")
                    pred_price = None
                if pred_price is None:
                    st.error("Không thể dự báo do lỗi mô hình.")
                else:
                    delta = pred_price - current_price
                    delta_pct = (delta / current_price) * 100
                    c1, c2 = st.columns(2)
                    c1.metric("Giá hiện tại", f"${current_price:.2f}")
                    c2.metric("AI Dự báo", f"${pred_price:.2f}", f"{delta:.2f} ({delta_pct:.2f}%)")
                    if delta > 0:
                        st.success(f"Xu hướng: TĂNG TRƯỞNG (+{delta:.2f})")
                    else:
                        st.error(f"Xu hướng: SUY GIẢM ({delta:.2f})")

with tab3:
    img_path = f"reports/{ticker}_evaluation.png"
    if os.path.exists(img_path):
        st.image(img_path, caption=f"Backtest Result: {ticker}", use_container_width=True)
    else:
        st.info("Chưa có báo cáo đánh giá.")