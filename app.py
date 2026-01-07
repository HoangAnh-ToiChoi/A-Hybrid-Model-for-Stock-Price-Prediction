import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.graph_objects as go
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler

# cấu hình
st.set_page_config(page_title="Stock AI Predictor", layout="wide")

# tắt gpu tránh deadlock
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# adđ đường dẫn src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.data_ingest import get_realtime_data, add_technical_indicators
from src.config import TIME_STEP, FEATURE_COLUMNS

@st.cache_resource
def load_model_and_scaler(ticker):
    
    tflite_path = f"models/{ticker}.tflite"
    keras_path = f"models/{ticker}_final_model.h5"
    scaler_path = f"models/{ticker}_scaler_y.pkl"
    
    # load scaler
    if not os.path.exists(scaler_path):
        return None, None, f"❌ Không tìm thấy Scaler tại: {scaler_path}", None
    
    try:
        scaler_y = joblib.load(scaler_path)
    except Exception as e:
        return None, None, f"❌ Lỗi đọc file Scaler: {e}", None

    # load tflite
    if os.path.exists(tflite_path):
        try:
            # Dùng tf.lite.Interpreter c
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            print(f"✅ Loaded TFLite model for {ticker}")
            return interpreter, scaler_y, None, "tflite"
        except Exception as e:
            print(f"⚠️ Lỗi TFLite: {e}. Đang thử fallback sang Keras...")
    
    # 3. fallback keras (dự phòng)
    if os.path.exists(keras_path):
        try:
            # compile=False để load nhanh hơn
            model = tf.keras.models.load_model(keras_path, compile=False)
            print(f"✅ Loaded Keras model for {ticker}")
            return model, scaler_y, None, "keras"
        except Exception as e:
            return None, None, f" Lỗi load Keras model: {e}", None
            
    return None, None, f" Không tìm thấy file model nào cho {ticker}", None


def run_model_inference(model, X_input: np.ndarray, mode: str) -> np.ndarray:
   
    if mode == "tflite":
        # Lấy thông tin input/output
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        idx = input_details[0]['index']
        X_input = X_input.astype(np.float32)
        
        # đưa dữ liệu vào
        model.set_tensor(idx, X_input)
        
        # chạy mô hình
        model.invoke()
        out_idx = output_details[0]['index']
        result = model.get_tensor(out_idx)
        return result
        
    elif mode == "keras":
        return model.predict(X_input, verbose=0)
    
    else:
        raise ValueError("không xác nhận được model")
def get_prediction_input(df, time_step):
    # lấy cột feature
    df_filtered = df[FEATURE_COLUMNS]
    
    #fit scaler để lấy min max
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_filtered.values)
    # lấy 100 dòng cuối
    last_100_days = df_filtered.values[-time_step:]
    #reshape
    scaled_data = scaler.transform(last_100_days)
    X_input = scaled_data.reshape(1, time_step, scaled_data.shape[1])
    
    return X_input.astype(np.float32)

st.sidebar.header("CẤU HÌNH")
ticker = st.sidebar.selectbox("Chọn mã cổ phiếu:", ["AAPL", "TSLA"])
years = st.sidebar.slider("Dữ liệu quá khứ (Năm):", 1, 5, 2)

st.title(f" Dự báo Chứng khoán: {ticker}")

# Tải dữ liệu (Có kiểm tra lỗi)
try:
    with st.spinner("Đang tải dữ liệu thị trường..."):
        df = get_realtime_data(ticker, years=years)
        df = add_technical_indicators(df)
        
        
        if df is None or df.empty:
            st.error(" Không tải được dữ liệu (Data Empty). Vui lòng kiểm tra mạng hoặc thử lại sau.")
            st.stop()

except Exception as e:
    st.error(f"Lỗi hệ thống: {e}")
    st.stop()

#load model 
model, scaler_y, model_err, model_mode = load_model_and_scaler(ticker)
tab1, tab2, tab3 = st.tabs([" Thị trường", " Dự báo", " Báo cáo"])

with tab1:
    current_price = df['Close'].iloc[-1]
    last_date = df.index[-1].strftime('%d-%m-%Y')
    
    # Tính biến động giá
    if len(df) > 1:
        change = current_price - df['Close'].iloc[-2]
        pct_change = (change / df['Close'].iloc[-2]) * 100
    else:
        change, pct_change = 0, 0

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
    
    if st.button("RUN", type="primary", use_container_width=True):
        if model is None or scaler_y is None:
            st.error(model_err)
        else:
            with st.spinner(f"Mô hình đang tính toán ({model_mode.upper()})..."):
                try:    
                    X_input = get_prediction_input(df, TIME_STEP)
                    
                    pred_scaled = run_model_inference(model, X_input, model_mode)
                    
                    pred_arr = np.array(pred_scaled).reshape(-1, 1)
                    pred_price = scaler_y.inverse_transform(pred_arr)[0][0]
                    
                    delta = pred_price - current_price
                    delta_pct = (delta / current_price) * 100
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Giá hiện tại", f"${current_price:.2f}")
                    c2.metric("AI Dự báo", f"${pred_price:.2f}", f"{delta:.2f} ({delta_pct:.2f}%)")
                    
                    if delta > 0:
                        st.success(f"Xu hướng: TĂNG TRƯỞNG (+{delta:.2f})")
                    else:
                        st.error(f"Xu hướng: SUY GIẢM ({delta:.2f})")
                        
                except Exception as e:
                    st.error(f"Lỗi khi chạy dự báo: {e}")

with tab3:
    img_path = f"reports/{ticker}_evaluation.png"
    if os.path.exists(img_path):
        st.image(img_path, caption=f"Backtest Result: {ticker}", use_container_width=True)
    else:
        st.info("Chưa có báo cáo đánh giá.")