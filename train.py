import pandas as pd
from xgboost import XGBRegressor
import joblib
import numpy as np

print("⏳ Đang tạo dữ liệu huấn luyện chuẩn 17 biến...")
np.random.seed(42)
n = 1500  

thang = np.random.randint(1, 13, n)
nam = np.random.randint(2020, 2027, n)
nhiet_do = np.random.uniform(20, 40, n)
do_am = np.random.uniform(50, 90, n)
mat_do_dan_so = np.random.uniform(100, 2500, n)
so_khach_hang = np.random.randint(10000, 60000, n)
toc_do_phat_trien = np.random.uniform(0.5, 10.0, n)
so_ngay_bao = np.random.randint(0, 5, n)

cup_dien_tuan = np.random.randint(0, 5, n)
cup_dien_cuoi_tuan = np.random.randint(0, 3, n)
ngay_le = np.random.randint(0, 5, n)
ngay_nghi = np.random.randint(8, 11, n)

pt1 = np.random.uniform(100000, 1500000, n)
pt2 = np.random.uniform(100000, 1500000, n)
pt3 = np.random.uniform(100000, 500000, n)
pt4 = np.random.uniform(3000000, 6000000, n)
pt5 = np.random.uniform(100000, 500000, n)

df = pd.DataFrame({
    'Thang': thang, 'Nam': nam, 'Nhiet_do': nhiet_do, 'Do_am': do_am,
    'Mat_do_dan_so': mat_do_dan_so, 'So_khach_hang': so_khach_hang, 'Toc_do_phat_trien': toc_do_phat_trien,
    'So_ngay_bao': so_ngay_bao, 'Cup_dien_tuan': cup_dien_tuan, 
    'Cup_dien_cuoi_tuan': cup_dien_cuoi_tuan, 'Ngay_le': ngay_le, 'Ngay_nghi': ngay_nghi,
    'Phu_tai_1': pt1, 'Phu_tai_2': pt2, 'Phu_tai_3': pt3, 'Phu_tai_4': pt4, 'Phu_tai_5': pt5
})

# THIẾT LẬP QUY LUẬT TÍNH TOÁN (GIỮ NGUYÊN YÊU CẦU CỦA BẠN)
tong_co_ban = pt1 + pt2 + pt3 + pt4 + pt5
tac_dong_nhiet = (nhiet_do - 28.0) * 50000
tac_dong_bao = so_ngay_bao * (-90000)
tac_dong_cup_tuan = cup_dien_tuan * (-90000)
tac_dong_cup_cuoituan = cup_dien_cuoi_tuan * (-60000)
tac_dong_le = ngay_le * (-30000)
tac_dong_nghi = ngay_nghi * (-30000)

df['San_luong_tong'] = tong_co_ban + tac_dong_nhiet + tac_dong_bao + tac_dong_cup_tuan + tac_dong_cup_cuoituan + tac_dong_le + tac_dong_nghi

X = df[['Thang', 'Nam', 'Nhiet_do', 'Do_am', 'Mat_do_dan_so', 'So_khach_hang', 'Toc_do_phat_trien', 'So_ngay_bao', 'Cup_dien_tuan', 'Cup_dien_cuoi_tuan', 'Ngay_le', 'Ngay_nghi', 'Phu_tai_1', 'Phu_tai_2', 'Phu_tai_3', 'Phu_tai_4', 'Phu_tai_5']]
y = df['San_luong_tong']

print("🧠 Đang huấn luyện bộ não AI...")
model = XGBRegressor(n_estimators=250, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X, y)

joblib.dump(model, 'xgboost_model.pkl')
print("✅ Xuất thành công file 'xgboost_model.pkl' mới!")