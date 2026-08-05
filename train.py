import pandas as pd
from xgboost import XGBRegressor
import joblib

print("Đang đọc dữ liệu lịch sử...")
# 1. Đọc dữ liệu huấn luyện từ file Excel thật
df = pd.read_excel('Du_lieu_lich_su.xlsx')

# DÒNG MỚI THÊM VÀO: Tự động xóa tất cả các hàng có chứa ô trống
df = df.dropna()

# Lọc bỏ các cột không dùng để huấn luyện (như STT) nếu có
if 'STT' in df.columns:
    df = df.drop(columns=['STT'])

# 2. Tách dữ liệu thành Biến đầu vào (X) và Biến mục tiêu (y)
# Giả sử cột chứa kết quả sản lượng tổng thực tế của bạn tên là 'San_luong_thuc_te'
X = df.drop(columns=['San_luong_thuc_te'])
y = df['San_luong_thuc_te']

print("Đang tiến hành huấn luyện mô hình XGBoost...")
# 3. Khởi tạo và huấn luyện mô hình AI
# Bạn có thể tinh chỉnh các thông số n_estimators, learning_rate sau này để AI thông minh hơn
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X, y)

# 4. Ghi đè file bộ não AI mới
joblib.dump(model, 'xgboost_model.pkl')
print("✅ Đã huấn luyện xong bằng dữ liệu THẬT và lưu vào file xgboost_model.pkl!")