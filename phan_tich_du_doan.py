import json

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Lưu danh sách chứa tên các quốc gia không bị trùng lặp
from sklearn.metrics import mean_squared_error, r2_score


def country_list_gen(df):
    # Đổi tên cột từ Country Name -> country_name
    df.rename(columns={'Country Name': 'country_name'}, inplace=True)

    # Chuyển tất cả giá trị của hàng country_name thành chữ in thường
    df['country_name'] = df['country_name'].apply(lambda row: row.lower())

    # Kiểm tra và tạo ra 1 danh sách chứa các tên quốc gia không bị trùng lặp
    _lists = df['country_name'].unique().tolist()

    # Tiến hành lưu danh sách tên quốc gia ra tệp json
    with open('dataset/country_list.json', 'w', encoding='utf-8') as f:
        json.dump(_lists, f, ensure_ascii=False, indent=4)

    return _lists, df


def describe(df):
    # Chỉ chọn những cột dữ liệu là kiểu số
    df = df.select_dtypes(include=['number'])

    print('[THỐNG KÊ MÔ TẢ BỘ DỮ LIỆU]')
    print(df.describe().to_string())


# Chọn hàng có quốc gia tương ứng với quốc gia nhập vào
def selecting_country(df, country):
    # Lấy hàng có country_name giống với country nhập vào
    df = df.loc[df['country_name'] == country]

    # Xóa cột country_name để thuận tiện cho việc xây dựng mô hình hồi quy tuyến tính
    df = df.drop(columns=['country_name'])

    # Chuyển vị => thu được ma trận 2 cột (cột 1 là năm, cột 2 là dân số theo năm)
    df = df.T

    # Đổi tên cột
    df = df.reset_index().rename(columns={255: 'population', 'index': 'year'})

    return df


# Xây dựng và huấn luyện với mô hình hồi quy tuyến tính
def prediction_model(df, country):
    # Sau khi selecting_country thì ta nhận được 1 ma trận có 2 cột (cột 1 là năm, cột 2 là dân số theo năm)
    # Thì giá trị cột 1 (cột năm) chính là biến độc lập (x)
    x = df.iloc[:, 0].values.reshape(-1, 1)  # Lấy toàn bộ hàng của cột 0 và chuyển nó thành mảng 2 chiều

    # Giá trị của cột 2 (cột dân số theo năm) sẽ là biến phụ thuộc (y)
    y = df.iloc[:, 1].values.reshape(-1, 1)  # Lấy toàn bộ hàng của cột 1 và chuyển nó thành mảng 2 chiều

    # Vẽ biểu đồ cột thể hiển dân số của quốc gia theo từng năm
    plt.figure(figsize=(10, 6))
    plt.bar(x.reshape(-1), y.reshape(-1), color='skyblue')
    plt.title(f'Dân số {country.upper()} theo từng năm')
    plt.xlabel('Năm')
    plt.ylabel('Dân số')
    plt.xticks(x.reshape(-1), rotation='vertical')
    plt.show()

    # Xây dựng mô hình hồi quy tuyến tính
    model = LinearRegression()

    # Tiến hành huấn luyện
    model.fit(x, y)

    # Đánh giá mô hình
    y_pred = model.predict(x)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print('\n[ĐÁNH GIÁ MÔ HÌNH]')
    print(f"Trung bình sai số (MSE): {mse:.2f}")
    print(f"Bình phương sai số (R²): {r2:.2f}")

    # Vẽ biểu đồ thể hiện đường hồi quy tuyến tính
    plt.figure(figsize=(10, 6))
    plt.scatter(x.reshape(-1), y.reshape(-1), color='red', marker='x', label='Dân số')
    plt.plot(x.reshape(-1), y_pred, color='blue', label='Phương trình đường thằng hồi quy tuyến tính')
    plt.title('Đồ thị hồi quy tuyến tính')
    plt.xlabel('Năm')
    plt.ylabel('Dân số')
    plt.xticks(x.reshape(-1), rotation='vertical')
    plt.legend()
    plt.show()

    return model


# Tính toán và trả về giá trị dự đoán dựa vào mô hình đã được huấn luyện
def prediction(model, year):
    return int(model.coef_[0][0] * year + model.intercept_[0])


# Đọc dữ liệu
df = pd.read_csv('dataset/new_pop.csv')

# Trích xuất tên quốc gia từ cột Country Name
lists, df = country_list_gen(df)

# Thống kê mô tả bộ dữ liệu
describe(df)

country = input("\nNhập vào tên quốc gia (nhập không dấu): ").lower()
year = int(input("Nhập vào năm muốn dự đoán: "))

# Kiểm tra tên quốc gia nhập vào có tồn tại trong lists không
if country in lists:
    # Nếu tồn tại thì sẽ tiến hành lấy hàng dữ liệu tương ứng theo quốc gia ấy
    df = selecting_country(df, country)

    # Sau đó tiến hành xây dựng và huấn luyện
    model = prediction_model(df, country)

    # Hàm prediction sẽ trả về kết quả dự đoán dân số quốc gia ấy theo năm được nhập vào
    result = prediction(model, year)
    print(f"\nKết quả: Dự đoán dân số {country.upper()} năm {year} sẽ là: {result:,d}")
else:
    print('Tên quốc gia không đúng, vui lòng kiểm tra lại tên quốc gia trong file country_list.json')
