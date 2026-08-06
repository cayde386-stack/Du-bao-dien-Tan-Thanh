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
    try:
        global model
        file = request.files.get('file')
        if not file:
            return jsonify({'status': 'error', 'message': 'Vui lòng chọn file Excel!'})
        
        df = pd.read_excel(file)
        
        results = []
        for index, row in df.iterrows():
            # Đọc các thông số cơ bản, nếu thiếu tự động gán giá trị mặc định an toàn
            thang_val = float(row.get('Thang', 7))
            nam_val = float(row.get('Nam', 2026))
            nhiet_do = float(row.get('Nhiet_do', 35))
            do_am = float(row.get('Do_am', 70))
            mat_do = float(row.get('Mat_do_dan_so', 200))
            khach_hang = float(row.get('So_khach_hang', 26000))
            toc_do = float(row.get('Toc_do_phat_trien', 0.8))
            
            thang_nang_nong = 1 if (3 <= thang_val <= 5) else 0
            thang_mua = 1 if (5 <= thang_val <= 11) else 0
            thang_bao = float(row.get('Thang_bao', row.get('So_ngay_bao', 0)))
            
            cup_dien_tuan = float(row.get('Cup_dien_tuan', 0))
            cup_dien_cuoi_tuan = float(row.get('Cup_dien_cuoi_tuan', 0))
            ngay_le = float(row.get('Ngay_le', 0))
            ngay_nghi = float(row.get('Ngay_nghi', 8))
            
            # Xử lý phụ tải cấu thành (nếu dòng nào để trống hoặc NaN thì tự động gán bằng 0 hoặc lấy giá trị lịch sử)
            pt1 = float(row.get('Phu_tai_1', 0) or 0)
            pt2 = float(row.get('Phu_tai_2', 0) or 0)
            pt3 = float(row.get('Phu_tai_3', 0) or 0)
            pt4 = float(row.get('Phu_tai_4', 0) or 0)
            pt5 = float(row.get('Phu_tai_5', 0) or 0)
            
            # Xử lý dữ liệu lịch sử cùng kỳ năm trước (nếu trống thì tự lấy bằng giá trị hiện tại để bỏ qua chênh lệch)
            pt1_ky = float(row.get('Pt1_ky_truoc', 0) or pt1)
            pt2_ky = float(row.get('Pt2_ky_truoc', 0) or pt2)
            pt3_ky = float(row.get('Pt3_ky_truoc', 0) or pt3)
            pt4_ky = float(row.get('Pt4_ky_truoc', 0) or pt4)
            pt5_ky = float(row.get('Pt5_ky_truoc', 0) or pt5)
            
            # Tạo DataFrame 24 biến cho từng dòng
            df_row = pd.DataFrame([{
                'Thang': thang_val, 'Nam': nam_val,
                'Nhiet_do': nhiet_do, 'Do_am': do_am,
                'Mat_do_dan_so': mat_do, 'So_khach_hang': khach_hang,
                'Toc_do_phat_trien': toc_do, 'Thang_nang_nong': thang_nang_nong,
                'Thang_mua': thang_mua, 'Thang_bao': thang_bao,
                'Cup_dien_tuan': cup_dien_tuan, 'Cup_dien_cuoi_tuan': cup_dien_cuoi_tuan,
                'Ngay_le': ngay_le, 'Ngay_nghi': ngay_nghi,
                'Phu_tai_1': pt1, 'Phu_tai_2': pt2,
                'Phu_tai_3': pt3, 'Phu_tai_4': pt4, 'Phu_tai_5': pt5,
                'Pt1_ky_truoc': pt1_ky, 'Pt2_ky_truoc': pt2_ky,
                'Pt3_ky_truoc': pt3_ky, 'Pt4_ky_truoc': pt4_ky,
                'Pt5_ky_truoc': pt5_ky
            }])
            
            # Thực hiện dự báo qua mô hình AI XGBoost
            pred = model.predict(df_row)[0]
            row['Ket_qua_du_bao'] = round(float(pred), 2)
            results.append(row)
            
        df_result = pd.DataFrame(results)
        html_table = df_result.to_html(classes='table table-striped table-bordered text-center', index=False)
        
        return jsonify({'status': 'success', 'html_table': html_table})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})