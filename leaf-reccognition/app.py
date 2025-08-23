from flask import Flask, request, jsonify
import os
import uuid
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/pest', methods=['POST'])
def suggestion():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # 保存上传的图片
    file = request.files['image']
    filename = f"{uuid.uuid4().hex}.jpg"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    # 调用 Suggestion.py
    command = [
        'python', 'Suggestion.py',
        '--img', input_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    # 获取 Suggestion.py 输出
    suggest_log = result.stdout.strip()

    return jsonify({
        'log': suggest_log
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
