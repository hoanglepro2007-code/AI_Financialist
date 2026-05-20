import streamlit as st 
import pandas as pd 
import numpy as np
import joblib
from Data_Mining import *
import plotly.graph_objects as go
import json
from tensorflow.keras.models import load_model
from supabase import create_client,Client
st.set_page_config(page_title="Stock market index report", page_icon=":signal_strength:")

st.title(":signal_strength: Stock market index report")


selected_ticker = st.sidebar.text_input("Nhập mã cổ phiếu (VD: VNM.VN):", value="HPG.VN").upper()
ticker = StockDataPipeline(selected_ticker)



@st.cache_data
def load_cache_data(ticker_name):
    # 1. Cấu hình chìa khóa kết nối (Giống y hệt bên Data_Mining.py)
    url = "https://kdysvuabyzbwpjbvazli.supabase.co" 
    project_api = "sb_publishable_kboslBcsF1dKvawyy_IXSw_Ou5h-L95"
    supabase: Client = create_client(url, project_api)
    
    try:
        # 2. Truy vấn dữ liệu từ bảng stock_prices lọc theo mã cổ phiếu, sắp xếp theo ngày tăng dần
        response = supabase.table('stock_prices') \
                           .select('*') \
                           .eq('ticker', ticker_name) \
                           .order('date', desc=True) \
                           .execute()
        
        # 3. Kiểm tra nếu không có dữ liệu trả về
        if not response.data:
            return None
            
        # 4. Ép dữ liệu JSON trả về thành DataFrame Pandas để các hàm vẽ biểu đồ bên dưới chạy mượt mà
        df = pd.DataFrame(response.data)
        
        # 5. Định dạng lại cột ngày và đưa về làm Index giống hệt cấu trúc cũ của em
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # 6. Đổi các cột tên viết thường trong SQL thành viết hoa chữ cái đầu để khớp với code vẽ biểu đồ Plotly phía dưới
        df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume',
            'ma50': 'MA50', 'ma20': 'MA20', 'rsi': 'RSI', 'macd': 'MACD', 
            'signal_line': 'Signal_Line', 'signal': 'Signal'
        }, inplace=True)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Lỗi khi lấy dữ liệu từ Supabase: {e}")
        return None

with st.spinner(f"🔄 Đang đồng bộ dữ liệu của mã {selected_ticker} từ Supabase..."):
    clean_data = load_cache_data(selected_ticker)

# Kiểm tra nếu chưa có dữ liệu trong Database
if clean_data is None or clean_data.empty:
    st.warning(f"⚠️ Mã cổ phiếu {selected_ticker} chưa có dữ liệu trên Hệ thống Đám mây Supabase.")
    st.info("💡 Vui lòng chạy file `AI_Agent.py` dưới máy cục bộ của em để đồng bộ dữ liệu của mã này lên mạng trước nhé!")
else:
    df = clean_data.copy()
    
    # Ép kiểu dữ liệu số để an toàn tuyệt đối khi vẽ biểu đồ Plotly
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA20', 'RSI', 'MACD', 'Signal_Line']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Sắp xếp lại theo thời gian để đường vẽ không bị rối
    df.sort_index(inplace=True)

    # ==========================================
    # BƯỚC 2: HIỂN THỊ TÍN HIỆU CẢNH BÁO TỪ AI (Lấy trực tiếp từ DB)
    # ==========================================
    st.subheader("🤖 Tín hiệu Khuyến nghị từ AI Agent")
    if 'Signal' in df.columns and not df['Signal'].empty:
        latest_signal = df['Signal'].iloc[-1]  # Lấy dòng tín hiệu mới nhất của ngày hôm nay
        latest_date = df.index[-1].strftime('%d/%m/%Y')
        
        if "BUY" in str(latest_signal).upper():
            st.success(f"🟩 **AI Khuyến nghị vào ngày {latest_date}:** {latest_signal}")
        elif "CARE" in str(latest_signal).upper() or "SELL" in str(latest_signal).upper():
            st.error(f"🟥 **AI Cảnh báo vào ngày {latest_date}:** {latest_signal}")
        else:
            st.info(f"🟦 **Trạng thái AI vào ngày {latest_date}:** {latest_signal}")
    else:
        st.info("ℹ️ Không tìm thấy cột tín hiệu khuyến nghị trong cơ sở dữ liệu.")

    # ==========================================
    # BƯỚC 3: VẼ BIỂU ĐỒ HOÀN TOÀN TỪ DỮ LIỆU ĐÁM MÂY
    # ==========================================
    st.subheader("📈 Biểu đồ phân tích kỹ thuật")

    # 1. Biểu đồ nến và các đường MA
    fig_candle = go.Figure()
    fig_candle.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Giá nến'
    ))
    if 'MA50' in df.columns:
        fig_candle.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA50', line=dict(color='blue')))
    if 'MA20' in df.columns:
        fig_candle.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20', line=dict(color='orange')))
        
    fig_candle.update_layout(title=f"Biểu đồ giá cổ phiếu {selected_ticker}", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_candle, use_container_width=True)

    # 2. Biểu đồ chỉ báo RSI
    if 'RSI' in df.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
        fig_rsi.add_shape(type="line", x0=df.index[0], y0=70, x1=df.index[-1], y1=70, line=dict(color="red", dash="dash"))
        fig_rsi.add_shape(type="line", x0=df.index[0], y0=30, x1=df.index[-1], y1=30, line=dict(color="green", dash="dash"))
        fig_rsi.update_layout(title="Chỉ số sức mạnh tương đối (RSI)", yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_rsi, use_container_width=True)

    # 3. Biểu đồ MACD
    if 'MACD' in df.columns and 'Signal_Line' in df.columns:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='red')))
        
        df['Histogram'] = df['MACD'] - df['Signal_Line']
        fig_macd.add_trace(go.Bar(
            x=df.index, y=df['Histogram'],
            marker_color=np.where(df['Histogram'] > 0, 'green', 'red'), name='Histogram'
        ))
        fig_macd.update_layout(title="Chỉ báo MACD")
        st.plotly_chart(fig_macd, use_container_width=True)
# if clean_data is None: 
#     st.warning(f"⚠️ Mã cổ phiếu {selected_ticker} chưa được cập nhật dữ liệu trên Hệ thống Đám mây Supabase. Vui lòng chạy file AI_Agent.py dưới máy cục bộ để đồng bộ!")
# else:
#     try:
#         with open(f'metrics/{selected_ticker}_metrics.json') as f:
#             metrics_data = json.load(f)
        

#         model_linear = joblib.load(f'models/{selected_ticker}_linear_model.pkl')

#         scaler_model_linear= joblib.load(f'models/{selected_ticker}_scaler.pkl')
    

    
#         with st.spinner(f"Data is being synchronized for year 5{selected_ticker}"):
        
#             clean_data = load_cache_data(selected_ticker)

#             bot = Important_Index(clean_data,selected_ticker)

#             df = bot.calculate_indicator()

    
        
#         X_latest = df[['MA50','MA20','RS','RSI','Volume','MACD','Signal_Line']].iloc[-1:]

#         X_latest_scaled = scaler_model_linear.transform(X_latest)

#         predicted_price = model_linear.predict(X_latest_scaled)[0]
        
#         predicted_price_ltsm = None
        
#         # try:
#         #     ltsm_model = load_model(f'models/{selected_ticker}_LTSM.keras')
#         #     sc_x = joblib.load(f'models/{selected_ticker}_scaler_x.pkl')
#         #     sc_y = joblib.load(f'models/{selected_ticker}_scaler_y.pkl')
            
#         #     x_60 = df[['MA50','MA20','RS','RSI','Volume','MACD','Signal_Line']].iloc[-60:].values
#         #     x_60_scaled = sc_x.transform(x_60)
#         #     x_3d = np.reshape(x_60_scaled,(1,60,7))
            
            
#         #     pred_scaled = ltsm_model.predict(x_3d)
#         #     predicted_price_ltsm = sc_y.inverse_transform(pred_scaled)[0][0]
#         # except Exception as e:
#         #     st.info("💡 Mô hình Deep Learning (LTSM) hiện chưa khả dụng cho mã này.")  
            
            
#         st.subheader("🎯 Kết Quả Phân Tích AI")

#         col1,col2,col3,col4 = st.columns(4) 


#         with col1:
#             display_price = predicted_price_ltsm if predicted_price_ltsm else predicted_price
#             label_text = "Predicted (LSTM)" if predicted_price_ltsm else "Predicted (Linear)"
#             st.metric(label=label_text, value=f"{display_price:,.2f}")
#         with col2:
#             st.metric(label="R-Squared",value=f"{metrics_data['R2']:,.2f}")
#         with col3:
#             st.metric(label="Accuracy rate",value=f"{metrics_data['Accuracy']*100:.1f}%")
#         with col4:
#             if df['RSI'].iloc[-1] >70 :
#                 st.text("Vùng Quá Mua (Overbought) - Rủi ro cao.")
#             elif df['RSI'].iloc[-1] <30:
#                 st.text("Vùng Quá Bán (Oversold) - Cơ hội mua.")
#             else:
#                 st.text("Trung Tính.")

#         # st.metric(label=f"Predict price tomorow: {selected_ticker}", value=f"{predicted_price: ,.2f}")

#         clean_data=clean_data.reset_index()

#         # Init Candle Stick Chart
#         candles_stick_chart = go.Candlestick(
#             x=clean_data['Date'],
#             open=clean_data['Open'],
#             high=clean_data['High'],
#             low=clean_data['Low'],
#             close=clean_data['Close']
            
#         )

#         # Figure this chart

#         fig = go.Figure(data=[candles_stick_chart])


#         # Use Streamlit to show chart on web

#         st.title(f"Candlestick chart of {selected_ticker}")
#         # Tham số use_container_width=True giúp biểu đồ tự động co giãn vừa với màn hình web.




#         # Add MA50


#         fig.add_trace(go.Scatter(
#             x=df.index,
#             y=df['MA50'],
#             mode='lines',
#             line=dict(color='purple'),
#             name= 'MA50'
            
            
#         ))

#         # Add MA20 
#         fig.add_trace(go.Scatter(
#             x=df.index,
#             y=df['MA20'],
#             mode='lines',
#             line=dict(color='red'),
#             name= 'MA20'
            
            
#         ))
#         fig_macd = go.Figure()
        
        
        

#         # SỬ DỤNG UPDATE_LAYOUT() ĐỂ TẮT THANH TRƯỢT

#         fig.update_layout(
#             xaxis_rangeslider_visible=False
#         )
#         st.plotly_chart(fig, use_container_width=True)
#         fig_macd.add_trace(go.Scatter(
#             x=df.index,
#             y=df['MACD'],
#             mode='lines',
#             line=dict(color='green'),
#             name='MACD'
#         ))
#         fig_macd.add_trace(go.Scatter(
#             x=df.index,
#             y=df['Signal_Line'],
#             mode='lines',
#             line=dict(color='yellow'),
#             name='Signal_Line'
#         ))
        
#         df['Histogram'] = df['MACD'] - df['Signal_Line']
#         fig_macd.add_trace(go.Bar(
#             x=df.index,
#             y=df['Histogram'],
#             marker_color=np.where(df['Histogram'] > 0, 'green', 'red'),
#             name='Histogram'
#         ))
#         st.plotly_chart(fig_macd, use_container_width=True)
#     except FileNotFoundError:
#         st.warning(f"⚠️ Mã cổ phiếu {selected_ticker} chưa được huấn luyện AI. Vui lòng chạy hệ thống AI_Agent để cập nhật dữ liệu cho mã này!")
#         if st.button("Huấn luyện AI cho mã này"):
#             st.spinner("Đang huấn luyện bộ não mới... Chờ xíu nhé!")
#             raw_data = ticker.fetch_data(period="5y", interval="1d")
        
#             if raw_data is not None:
#                 clean_data = ticker.clean_data(raw_data)
#                 important_index_bot = Important_Index(clean_data, selected_ticker)
                
#                 # Chạy toàn bộ quy trình như cũ
#                 important_index_bot.calculate_indicator()
#                 important_index_bot.calculate_Volatility()
#                 important_index_bot.train_and_save_model() 
#                 important_index_bot.train_and_save_model() 
#                 important_index_bot.train_LTSM()
                
#                 final_data = important_index_bot.alert_system()
#                 ticker.save_data(final_data)
#                 important_index_bot.plot_chart(selected_ticker)
#                 st.rerun()
#             else:
#                 st.error("Không thể lấy dữ liệu từ Yahoo Finance. Kiểm tra lại mã cổ phiếu!")
#     except Exception as e:
#         st.error(f"The system experienced a data connection interruption. Please try again later! (Error code: {e})")

