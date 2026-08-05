from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import io

app = Flask(__name__)

# Tải mô hình AI
def load_model():
    try:
        return joblib.load('xgboost_model.pkl')
    except:
        return None

model = load_model()

@app.route('/')
def home():
    return render_template('index.html', title="Dự báo Điện Thương Phẩm")

@app.route('/predict', methods=['POST'])
def predict_manual():
    global model
    if model is None:
        return jsonify({'status': 'error', 'message': 'Chưa có mô hình AI 13 biến. Vui lòng Huấn luyện trước.'})
    try:
        data = request.json
        df_manual = pd.DataFrame([{
            'Thang': float(data['thang']), 'Nam': float(data['nam']),
            'Nhiet_do': float(data['nhiet_do']), 'Do_am': float(data['do_am']),
            'Mat_do_dan_so': float(data['mat_do']), 'So_khach_hang': float(data['khach_hang']),
            'Toc_do_phat_trien': float(data['toc_do']), 'Co_bao': float(data['bao']),
            'Phu_tai_1': float(data['pt1']), 'Phu_tai_2': float(data['pt2']),
            'Phu_tai_3': float(data['pt3']), 'Phu_tai_4': float(data['pt4']), 'Phu_tai_5': float(data['pt5'])
        }])
        
        prediction = model.predict(df_manual)[0]
        return jsonify({'status': 'success', 'result': "{:,.2f}".format(prediction)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/download_template')
def download_template():
    # Tự động tạo file Excel mẫu
    columns = [
        'STT', 'Thang', 'Nam', 'Nhiet_do', 'Do_am', 'Mat_do_dan_so', 
        'So_khach_hang', 'Toc_do_phat_trien', 'Co_bao', 
        'Phu_tai_1', 'Phu_tai_2', 'Phu_tai_3', 'Phu_tai_4', 'Phu_tai_5'
    ]
    df = pd.DataFrame(columns=columns)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Du_lieu_nhap')
    output.seek(0)
    
    return send_file(output, download_name="File_Mau_Du_Bao.xlsx", as_attachment=True)

@app.route('/predict_excel', methods=['POST'])
def predict_excel():
    global model
    if model is None:
        return jsonify({'status': 'error', 'message': 'Chưa tìm thấy mô hình AI.'})
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'status': 'error', 'message': 'Chưa chọn file.'})

    try:
        df = pd.read_excel(file)
        if 'STT' in df.columns: df_for_ai = df.drop(columns=['STT'])
        else: df_for_ai = df.copy()

        df['Tổng Phụ Tải (Gốc)'] = df['Phu_tai_1'] + df['Phu_tai_2'] + df['Phu_tai_3'] + df['Phu_tai_4'] + df['Phu_tai_5']
        
        lr = LinearRegression()
        X_lr = df[['Nhiet_do', 'Do_am', 'So_khach_hang']]
        lr.fit(X_lr, df['Tổng Phụ Tải (Gốc)'])
        df['1. Hồi quy'] = lr.predict(X_lr).round(2)

        df['2. TB Động'] = df['Tổng Phụ Tải (Gốc)'].rolling(window=3, min_periods=1).mean().round(2)

        mean_load = df['Tổng Phụ Tải (Gốc)'].mean()
        std_load = df['Tổng Phụ Tải (Gốc)'].std()
        if pd.isna(std_load) or std_load == 0: std_load = 1
        df['Z-Score'] = ((df['Tổng Phụ Tải (Gốc)'] - mean_load) / std_load).round(2)
        df['3. Cảnh báo'] = np.where(df['Z-Score'] > 1.5, '⚠️ Tăng', np.where(df['Z-Score'] < -1.5, '⏬ Giảm', '✅ Ổn định'))
        
        df['4. AI XGBoost'] = model.predict(df_for_ai).round(2)
        df = df.drop(columns=['Z-Score', 'Tổng Phụ Tải (Gốc)'])
        
        html_table = df.to_html(classes='table table-bordered table-striped table-hover text-center align-middle', index=False)
        html_table = html_table.replace('⚠️ Tăng', '<span class="badge bg-danger">⚠️ Tăng</span>').replace('⏬ Giảm', '<span class="badge bg-warning text-dark">⏬ Giảm</span>').replace('✅ Ổn định', '<span class="badge bg-success">✅ Ổn định</span>')

        return jsonify({'status': 'success', 'html_table': html_table})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Lỗi dữ liệu đầu vào: Vui lòng sử dụng đúng File Mẫu.'})

if __name__ == '__main__':
    app.run(debug=True)
