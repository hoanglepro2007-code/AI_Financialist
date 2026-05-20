from Data_Mining import *
import logging

logging.basicConfig(
    filename='C:\\Users\\MSl\\Desktop\\AI Agent\\HISTORY_TERNIMAL\\history.log',      # Tên file lưu log
    level=logging.DEBUG,# Ghi lại tất cả từ mức DEBUG trở lên
     
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True)

if __name__ == "__main__":
    ticker_input = input("Nhập mã cổ phiếu muốn huấn luyện (VD: VNM.VN): ").upper()
    
    print(f"--- Đang bắt đầu quy trình huấn luyện cho {ticker_input} ---")
    ticker = StockDataPipeline(ticker_input) # Tên biến hpg giờ chỉ là đại diện
    
    raw_data = ticker.fetch_data(period="5y", interval="1d")
    
    if raw_data is not None:
        clean_data = ticker.clean_data(raw_data)
        important_index_bot = Important_Index(clean_data, ticker_input)
        
        # Quy trình tính toán chỉ số
        important_index_bot.calculate_indicator()
        important_index_bot.calculate_Volatility()
        important_index_bot.train_and_save_model() 
        important_index_bot.train_LTSM() # Đảm bảo chạy cả mô hình học sâu nhé
        
        final_data = important_index_bot.alert_system()
        
        # THAY THẾ ĐOẠN NÀY: Gọi hàm bắn lên Supabase thay vì lưu CSV
        important_index_bot.save_to_supabase(final_data)
        
        important_index_bot.plot_chart(ticker_input)
    
    
    

