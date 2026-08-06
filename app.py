from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import io

app = Flask(__name__)

def load_model():
    try:
        return joblib.load('xgboost_model.pkl')
    except:
        return None

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Xử lý an toàn dữ liệu đầu vào và các trường hợp bỏ trống
        thang_val = float(data.get('thang', 7))
        thang_nang_nong = 1 if (3 <= thang_val <= 5) else 0
        thang_mua = 1 if (5 <= thang_val <= 11) else 0
        thang_bao = float(data.get('so_ngay_bao') or 0)
        
        pt1 = float(data.get('pt1', 0))
        pt2 = float(data.get('pt2', 0))
        pt3 = float(data.get('pt3', 0))
        pt4 = float(data.get('pt4', 0))
        pt5 = float(data.get('pt5', 0))
        
        # Nếu không nhập dữ liệu lịch sử thì tự lấy bằng chính giá trị hiện tại để bỏ qua
        pt1_ky = float(data.get('pt1_ky_truoc') or pt1)
        pt2_ky = float(data.get('pt2_ky_truoc') or pt2)
        pt3_ky = float(data.get('pt3_ky_truoc') or pt3)
        pt4_ky = float(data.get('pt4_ky_truoc') or pt4)
        pt5_ky = float(data.get('pt5_ky_truoc') or pt5)

        df_manual = pd.DataFrame([{
            'Thang': thang_val, 
            'Nam': float(data.get('nam', 2026)),
            'Nhiet_do': float(data.get('nhiet_do', 0)), 
            'Do_am': float(data.get('do_am', 0)),
            'Mat_do_dan_so': float(data.get('mat_do', 0)), 
            'So_khach_hang': float(data.get('khach_hang', 0)),
            'Toc_do_phat_trien': float(data.get('toc_do', 0)), 
            'Thang_nang_nong': thang_nang_nong,
            'Thang_mua': thang_mua, 
            'Thang_bao': thang_bao,
            'Cup_dien_tuan': float(data.get('cup_dien_tuan', 0)), 
            'Cup_dien_cuoi_tuan': float(data.get('cup_dien_cuoi_tuan', 0)),
            'Ngay_le': float(data.get('ngay_le', 0)), 
            'Ngay_nghi': float(data.get('ngay_nghi', 8)),
            'Phu_tai_1': pt1, 
            'Phu_tai_2': pt2,
            'Phu_tai_3': pt3, 
            'Phu_tai_4': pt4, 
            'Phu_tai_5': pt5,
            'Pt1_ky_truoc': pt1_ky, 
            'Pt2_ky_truoc': pt2_ky,
            'Pt3_ky_truoc': pt3_ky, 
            'Pt4_ky_truoc': pt4_ky, 
            'Pt5_ky_truoc': pt5_ky
        }])

        prediction = model.predict(df_manual)[0]
        result = round(float(prediction), 2)
        
        return jsonify({'status': 'success', 'result': f"{result:,.2f}"})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/download_template')
def download_template():
    # Tạo DataFrame với đầy đủ các cột chuẩn hóa khớp với mô hình AI 24 biến
    df_template = pd.DataFrame({
        'STT': [1, 2],
        'Thang': [7, 8],
        'Nam': [2026, 2026],
        'Nhiet_do': [40.0, 38.5],
        'Do_am': [30.0, 35.0],
        'Mat_do_dan_so': [200, 200],
        'So_khach_hang': [26623, 26623],
        'Toc_do_phat_trien': [0.8, 0.8],
        'Thang_nang_nong': [0, 0],
        'Thang_mua': [1, 1],
        'Thang_bao': [2, 1],
        'Cup_dien_tuan': [0, 1],
        'Cup_dien_cuoi_tuan': [0, 0],
        'Ngay_le': [0, 0],
        'Ngay_nghi': [8, 9],
        'Phu_tai_1': [1450000, 1500000],
        'Phu_tai_2': [1350000, 1400000],
        'Phu_tai_3': [350000, 360000],
        'Phu_tai_4': [5250000, 5300000],
        'Phu_tai_5': [400000, 410000],
        'Pt1_ky_truoc': [1400000, 1450000],
        'Pt2_ky_truoc': [1300000, 1350000],
        'Pt3_ky_truoc': [340000, 350000],
        'Pt4_ky_truoc': [5100000, 5200000],
        'Pt5_ky_truoc': [390000, 400000]
    })
    
    file_path = 'File_Mau_Du_Bao.xlsx'
    df_template.to_excel(file_path, index=False)
    return send_file(file_path, as_attachment=True)
@app.route('/predict_excel', methods=['POST'])
def predict_excel():
    global model
    file = request.files.get('file')
    if not file: return jsonify({'status': 'error', 'message': 'Chưa chọn file.'})
    try:
        df = pd.read_excel(file)
        if 'STT' in df.columns: df_for_ai = df.drop(columns=['STT'])
        else: df_for_ai = df.copy()

        df['Tổng (Gốc)'] = df['Phu_tai_1'] + df['Phu_tai_2'] + df['Phu_tai_3'] + df['Phu_tai_4'] + df['Phu_tai_5']
        
        lr = LinearRegression()
        X_lr = df[['Nhiet_do', 'Do_am', 'So_khach_hang']]
        lr.fit(X_lr, df['Tổng (Gốc)'])
        df['1. Hồi quy'] = lr.predict(X_lr).round(2)
        df['2. TB Động'] = df['Tổng (Gốc)'].rolling(window=3, min_periods=1).mean().round(2)

        mean_load = df['Tổng (Gốc)'].mean()
        std_load = df['Tổng (Gốc)'].std()
        if pd.isna(std_load) or std_load == 0: std_load = 1
        df['Z-Score'] = ((df['Tổng (Gốc)'] - mean_load) / std_load).round(2)
        df['3. Cảnh báo'] = np.where(df['Z-Score'] > 1.5, '⚠️ Tăng', np.where(df['Z-Score'] < -1.5, '⏬ Giảm', '✅ Ổn định'))
        
        df['4. AI XGBoost'] = model.predict(df_for_ai).round(2)
        df = df.drop(columns=['Z-Score', 'Tổng (Gốc)'])
        
        html_table = df.to_html(classes='table table-bordered table-striped table-hover text-center align-middle', index=False)
        html_table = html_table.replace('⚠️ Tăng', '<span class="badge bg-danger">⚠️ Tăng</span>').replace('⏬ Giảm', '<span class="badge bg-warning text-dark">⏬ Giảm</span>').replace('✅ Ổn định', '<span class="badge bg-success">✅ Ổn định</span>')

        return jsonify({'status': 'success', 'html_table': html_table})
    except Exception as e:
        return jsonify({'status': 'error', 'message': 'Vui lòng tải lại File Mẫu mới chứa cột Lịch cúp điện!'})

if __name__ == '__main__':
    app.run(debug=True)