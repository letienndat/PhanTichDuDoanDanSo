import pandas as pd

# Đọc dữ liệu từ bộ pop.csv, bỏ qua 4 hàng đầu
df = pd.read_csv('dataset/pop.csv', skiprows=4)
print('[DỮ LIỆU NGUỒN]')
print(df)

# Xóa các cột dữ liệu nhiễu
df = df.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code'])
print('[DỮ LIỆU SAU KHI XÓA CỘT DỮ LIỆ NHIỄU]')
print(df)

# Kiểm tra giá trị khuyết
missing_values = df.isnull().sum()
print('[ĐẾM GIÁ TRỊ KHUYẾT CỦA MỖI CỘT]')
print(missing_values)

# Xóa hàng chứa giá trị khuyết
df_cleaned = df.dropna()
print('[DỮ LIỆU SAU KHI XÓA HÀNG CHỨA GIÁ TRỊ KHUYẾT]')
print(df_cleaned)

# Lưu file kết quả của quá trình tiền xử lý
df_cleaned.to_csv('dataset/new_pop.csv', index=False)
