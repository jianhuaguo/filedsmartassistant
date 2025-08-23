from flask import Flask, request, jsonify
import os
import uuid
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/fresh', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # 保存上传的图片
    file = request.files['image']
    filename = f"{uuid.uuid4().hex}.jpg"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    # 调用 predict.py
    command = [
        'python', 'src/predict.py',
        '--img', input_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    # 获取 predict.py 输出
    predict_log = result.stdout.strip()

    return jsonify({
        'log': predict_log
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
