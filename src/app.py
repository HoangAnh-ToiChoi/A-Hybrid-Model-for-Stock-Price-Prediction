import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go

# Import hàm xử lý từ file inference.py
from inference import get_latest_stock_data, prepare_input_for_prediction

# cấu hình
st.set_page_config(page_title="Stock AI Predictor", layout="wide")

st.title("Dự báo Thị trường Chứng khoán (CNN-LSTM)")
st.markdown("Hệ thống sử dụng mô hình Deep Learning để dự báo giá đóng cửa ngày tiếp theo.")

# sidebar:pick mã
st.sidebar.header("Tùy chọn")
ticker = st.sidebar.selectbox("Chọn mã cổ phiếu:", ["AAPL", "TSLA"])

# hiển thị gía lịch sử
st.subheader(f"Biểu đồ giá thực tế của {ticker}")

# Gọi hàm lấy dữ liệu live
df = get_latest_stock_data(ticker)

if df is not None:
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    
    fig.update_layout(height=400, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Hiển thị giá hôm nay
    current_price = float(df['Close'].iloc[-1])
    st.metric(label="Giá đóng cửa gần nhất", value=f"${current_price:.2f}")

    st.subheader("Dự báo tương lai")

    if st.button("Chạy mô hình dự đoán (Next Day)"):
        with st.spinner("AI đang tính toán... vui lòng chờ..."):
            # 1. Load Model
            model_path = f"models/{ticker}_model.h5"
            
            if not os.path.exists(model_path):
                st.error(f"Không tìm thấy file model: {model_path}. Hãy chạy train.py trước!")
            else:
                try:
                    # Load model
                    model = tf.keras.models.load_model(model_path)
                    
                    # 2. Chuẩn bị dữ liệu đầu vào
                    X_input, scaler, error_msg = prepare_input_for_prediction(ticker, df)
                    
                    if error_msg:
                        st.error(error_msg)
                    else:
                        # 3. Dự đoán
                        prediction_scaled = model.predict(X_input)
                        
                        # 4. Dịch ngược về giá USD
                        prediction_price = scaler.inverse_transform(prediction_scaled)
                        final_price = prediction_price[0][0]
                        
                        # 5. So sánh và hiển thị
                        delta = final_price - current_price
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Giá hiện tại", f"${current_price:.2f}")
                        col2.metric("Giá dự đoán ngày mai", f"${final_price:.2f}", f"{delta:.2f} USD")
                        
                        if delta > 0:
                            st.success("Tín hiệu: TĂNG TRƯỞNG ")
                        else:
                            st.warning("Tín hiệu: SUY GIẢM ")
                            
                except Exception as e:
                    st.error(f"Lỗi khi chạy model: {e}")

else:
    st.error("Không tải được dữ liệu từ Yahoo Finance.")
    st.stop()

#  Đánh giá hiệu năng(mới làm thêm)
st.write("---") 
st.subheader(" Đánh giá độ chính xác của Mô hình")

# Tạo 2 cột để hiển thị thông tin
tab1, tab2 = st.tabs(["Biểu đồ So sánh", "Số liệu chi tiết"])

with tab1:
    st.write(f"Kết quả kiểm thử trên tập dữ liệu quá khứ của {ticker}:")
    
    # Đường dẫn ảnh báo cáo (tự động lấy theo mã đang chọn)
    image_path = f"reports/{ticker}_evaluation.png"
    
    if os.path.exists(image_path):
        st.image(image_path, caption=f"So sánh Thực tế vs Dự đoán ({ticker})", use_container_width=True)
    else:
        st.warning(f" Chưa có báo cáo cho mã {ticker}. ")

with tab2:
    st.write("Các chỉ số đánh giá độ tin cậy:")
    if ticker == "AAPL":
        st.success("  Mô hình hoạt động RẤT TỐT trên Apple")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Độ chính xác (R2)", "88.61%", "Rất cao")
        col_b.metric("Sai số (RMSE)", "$7.05", "- thấp")
        col_c.metric("Sai số % (MAPE)", "3.83%", "Rất thấp")
    elif ticker == "TSLA":
        st.warning("  Mã Tesla có biến động mạnh, độ chính xác ở mức KHÁ")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Độ chính xác (R2)", "74.32%")
        col_b.metric("Sai số (RMSE)", "$28.42")
        col_c.metric("Sai số % (MAPE)", "9.56%")