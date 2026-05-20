import yfinance as yf
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as mlt
import os
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import math
import joblib
import json
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from supabase import create_client,Client







logging.warning("Warning: Something went wrong!") # Ghi lại cảnh báo vào file log
logging.error("Error: Can't connect to data!")# Ghi lại lỗi vào file log
logging.basicConfig(
    filename='HISTORY_TERNIMAL/history.log',      # Tên file lưu log
    level=logging.DEBUG,     # Ghi lại tất cả từ mức DEBUG trở lên
    format='%(asctime)s - %(levelname)s - %(message)s' ) # Định dạng: [Thời gian] - [Mức độ] - [Nội dung])


class StockDataPipeline:
    def __init__(self,ticker):
        self.ticker = ticker
    def fetch_data(self,period="5y",interval="1d"):
        try:
            data = yf.Ticker(self.ticker).history(period=period,interval=interval)
            
            if data.empty:
                raise ValueError(f"Ticker {self.ticker} no data!") # Hàm rasie để báo lỗi Logic xuống dưới cho khối lệnh Except chạy
            return data
            
        except Exception as e:
            logging.error(f"Detailed Error: {e}") # Hiển thị lỗi chi tiết trong file log
            return None
    def clean_data(self,data):
        data.info()
        data.isnull().sum() # Đếm xem có bao nhiêu cột ko có giá trị 
        
        
        
        data = data[['Open','High','Low','Close','Volume']].copy()
        data['Volume'] = data['Volume'] / 1_000_000
        data.ffill(inplace=True) # Dùng method "ffill" => Forward Fill lấy giá trị ở trước fill vào ô bị Nan
        data.dropna(inplace=True)
        return data
    def save_data(self,data):
        data.to_csv(f"CLEAN_DATA/{self.ticker}_clean_data.csv",index=True) # Lưu dữ liệu đã làm sạch vào file CSV, index=True để lưu cả chỉ số (index) vào file CSV
class Important_Index:
    def __init__(self,data_frame,ticker_symbol):
        self.df = data_frame.copy()
        self.symbol = ticker_symbol

    def calculate_Volatility(self):
        
         self.df['Volatility'] = self.df['Close'].pct_change().rolling(window=20,step=1).std() *100 # Tính độ biến động theo phần trăm
    def alert_system(self):
        
        self.df['Signal'] = np.where((self.df['Close'] < self.df['MA50']) & (self.df['Volatility']<2.0),"Buy!","Be Carefull!")
        return self.df
    def plot_chart(self,ticker_name):
        folder_path = ".n8n-files"
        file_path = f"{folder_path}/{ticker_name.replace('.VN', '')}_MA50_Graph.png"
        fig, ax1 = mlt.subplots(figsize=(16, 9))
        
        ax2 = ax1.twinx()
        mlt.grid(True)
        
        
        ax1.set_ylabel('Price(VND)')
        ax2.set_ylabel("Volatility(%)")
        mlt.gcf().autofmt_xdate()
        mlt.title("The chart shows the MA50 and Close")
        
        
        ax1.plot(self.df.index,self.df['MA50'],color='purple',label='MA50')
        ax1.plot(self.df.index,self.df['Close'],color='blue',label='Close')
        
        
        
        ax2.plot(self.df.index,self.df['Volatility'],color='red',linestyle='--',label='Volatility')
        
        handles1,labels1 = ax1.get_legend_handles_labels()
        handles2,labels2 = ax2.get_legend_handles_labels()
        
        
        
        ax1.legend(handles1+handles2,labels1+labels2,loc='best',frameon=False)
        
        
        mlt.tight_layout()
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        
        mlt.savefig(file_path)
        mlt.close()
        webhook_url = "http://localhost:5678/webhook-test/8c538c19-351e-45dd-9902-c860baf760ee"
        try:
                requests.post(webhook_url)
                print("Đã báo cáo cho n8n gửi ảnh!")
        except Exception as e:
                print(f"Lỗi gọi n8n: {e}")
    def calculate_indicator(self):
        
        #calculate_MA50(self):
        self.df['MA50'] = self.df['Close'].rolling(window=50).mean()

        #calculate_MA20(self):
        self.df['MA20'] = self.df['Close'].rolling(window=20).mean()
        
        
        self.df['Price_fluctuations'] =  self.df['Close'].diff()
        #Gain:
        self.df['Gain'] = self.df['Price_fluctuations'].where(self.df['Price_fluctuations']>0,0)
        #Loss 
        self.df["Loss"] = self.df['Price_fluctuations'].where(self.df['Price_fluctuations']<0,0).abs()
        #AVGgain:
        self.df['AVG_gain'] = self.df['Gain'].rolling(window=14).mean()
        #AVGloss:
        self.df['AVG_loss'] = self.df['Loss'].rolling(window=14).mean()
        #RS:
        self.df['RS'] = self.df['AVG_gain'] / (self.df['AVG_loss'] + 1e-10)
        #RSI
        self.df['RSI'] = 100 - (100/(1+self.df["RS"]))
        #Target
        self.df['Target'] = self.df['Close'].shift(-1)
        #EMA_12
        self.df['EMA_12'] = self.df['Close'].ewm(span=12,adjust=False).mean()
        #EMA_26
        self.df['EMA_26'] = self.df['Close'].ewm(span=26,adjust=False).mean()
        #MACD_Line
        self.df['MACD'] = self.df['EMA_12'] - self.df['EMA_26']
        #Signal Line
        self.df['Signal_Line'] = self.df['MACD'].ewm(span=9,adjust=False).mean()
        
        # Drop NA values after calculating indicators
        self.df.dropna(inplace=True)
        print(self.df.isnull().sum())
        
        
        return self.df
    def train_and_save_model(self):

        X = self.df[['MA50','MA20','RS','RSI','Volume','MACD','Signal_Line']]
        Y = self.df['Target']
        
        
        
        scaler = StandardScaler()
        
        
        
        split_point=int(len(self.df)*0.8)
        
        X_train = X.iloc[:split_point]
        x_train_scaled = scaler.fit_transform(X_train)
        X_test = X.iloc[split_point:]
        x_test_scaled = scaler.transform(X_test)
        Y_train = Y.iloc[:split_point]
        y_test = Y.iloc[split_point:]
        
        
        
        
        
        model = LinearRegression()
        
        
        model.fit(x_train_scaled,Y_train)
        y_pred = model.predict(x_test_scaled)
        mse = mean_squared_error(y_test,y_pred)
       
        rmse = math.sqrt(mse)
        rse = r2_score(y_test,y_pred)
        print("-------------")
        print(f"MSE:{mse}")
        print(f"RMSE:{rmse}")
        print(f"RSE:{rse}")
        print(self.df[['Close', 'Target']].head())
        
        
        test_data = self.df.iloc[split_point:].copy()
        
        test_data['y_pred'] = y_pred
        test_data['Test_Signal'] = np.where(test_data['y_pred'] > test_data['Close'], 1, 0)
        
        
        accuracy = (test_data['Test_Signal'] == (test_data['Target'] > test_data['Close'])).mean()
        print(f"Độ chính xác hướng đi: {accuracy * 100:.2f}%")
        
        

        
        lastest_data = X.iloc[-1:]
        lastest_data_scaler = scaler.transform(lastest_data)

        
        towmorow_price = model.predict(lastest_data_scaler)
        
        
        print(f"Tomorow price of {towmorow_price}")
        
        
        joblib.dump(model,f'models/{self.symbol}_linear_model.pkl')
        joblib.dump(scaler,f'models/{self.symbol}_scaler.pkl')
        
        metric_data = {
            "R2":rse,
            "Accuracy":accuracy
        }
        with open(f'metrics/{self.symbol}_metrics.json','w') as f :
            json.dump(metric_data,f,indent=3)
    def train_LTSM(self):
        features = self.df[['MA50','MA20','RS','RSI','Volume','MACD','Signal_Line']].values
        target = self.df['Close'].values.reshape(-1,1)
        
        # Khởi tạo Scaler
        scalar_x = MinMaxScaler(feature_range=(0,1))
        scalar_y = MinMaxScaler(feature_range=(0,1))
        
        # Biến đổi dữ liệu
        data_scaled = scalar_x.fit_transform(features)
        target_scaled = scalar_y.fit_transform(target)
        
        
        x_train = []
        y_train = []
        window_size = 60
        # Vòng lặp "trượt" từ ngày thứ 60 đến hết tập dữ liệ
        for i in range(window_size,len(data_scaled)):
            # Lấy 60 dòng dữ liệu trước đó (từ i-60 đến i) và tất cả 7 cột
            x_train.append(data_scaled[i-window_size:i,:])
            
            # Lấy giá trị mục tiêu của ngày hiện tại (ngày thứ i)
            y_train.append(target_scaled[i,0])
            
            
            
        x_train,y_train = np.array(x_train),np.array(y_train)
        
        # 1. Khởi tạo mô hình
        model = Sequential()
        
        # 2. Thêm lớp LSTM đầu tiên
        # input_shape=(60, 7) vì em có 60 ngày và 7 chỉ báo
        model.add(LSTM(units=50, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2])))
        
        # 3. Thêm lớp Dropout để chống học vẹt
        model.add(Dropout(0.2))
        
        
        # 4. Lớp đầu ra (Dự báo 1 mức giá)
        model.add(Dense(units=1))
        
        
        # 5. Cấu hình bộ chọn lọc
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        print("--- Đang bắt đầu huấn luyện Deep Learning (LSTM) trên GPU ---")
        train = model.fit(
            x_train,y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2, # Lấy 20% dữ liệu để làm bài test
            verbose = 1 # Hiển thị quá trình học ra màn hình
        )
        
        model.save(f'models/{self.symbol}_LTSM.keras')
        joblib.dump(scalar_x, f'models/{self.symbol}_scaler_x.pkl')
        joblib.dump(scalar_y, f'models/{self.symbol}_scaler_y.pkl')
    
    def save_to_supabase(self,data):
        url ="https://kdysvuabyzbwpjbvazli.supabase.co" 
        project_api = "sb_publishable_kboslBcsF1dKvawyy_IXSw_Ou5h-L95"
        supabase: Client = create_client(url,project_api)

        
        
        df_db = data.reset_index().copy()
        
        # 3. Chuẩn hóa tên cột và định dạng để khớp hoàn toàn với bảng SQL đã tạo
        records = []
        for _, row in df_db.iterrows():
            record = {
                "date": row['Date'].strftime('%Y-%m-%d'), # Chuyển Timestamp của Pandas thành chuỗi YYYY-MM-DD
                "ticker": self.symbol,
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": float(row['Volume']),
                "ma50": float(row['MA50']),
                "ma20": float(row['MA20']),
                "rsi": float(row['RSI']),
                "macd": float(row['MACD']),
                "signal_line": float(row['Signal_Line']),
                "signal": str(row['Signal'])
            }
            records.append(record)
            
        try:
            print(f"--- Đang đồng bộ {len(records)} dòng dữ liệu của {self.symbol} lên Supabase ---")
            
            # 4. Thực hiện UPSERT (Nếu trùng ngày + trùng mã thì cập nhật mới, nếu chưa có thì thêm vào)
            # Giúp không bị lỗi trùng lặp dữ liệu khi em chạy lại Agent nhiều lần
            # Sửa lại dòng upsert trong Data_Mining.py của em:
            response = supabase.table('stock_prices').upsert(records, on_conflict='date,ticker').execute()
            
            print("✅ Đồng bộ dữ liệu lên Supabase thành công!")
        except Exception as e:
            print(f"❌ Lỗi khi đẩy dữ liệu lên Supabase: {e}")
            
            
            
            
        