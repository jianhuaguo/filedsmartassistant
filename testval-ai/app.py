from flask import Flask, request, jsonify
import os
import uuid
import subprocess
import glob
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # 保存上传的图片
    file = request.files['image']
    filename = f"{uuid.uuid4().hex}.jpg"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    output_path = os.path.join(RESULT_FOLDER, filename)

    # 调用 YOLO detect.py，输出到 RESULT_FOLDER 目录
    command = [
        'python', 'src/detect.py',
        '--img', input_path,
        '--model', 'weights/best.pt',
        '--output', output_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    detect_log = result.stdout.strip()


    jpg_files = glob.glob(os.path.join(output_path, '**', '*.jpg'), recursive=True)

    if not jpg_files:
        raise FileNotFoundError("No output image found in YOLO output directory")

    # 取第一个检测后的图片
    detected_image_path = jpg_files[0]


    # 读取检测后的图片并编码为 Base64
    with open(detected_image_path, 'rb') as img_f:
        img_bytes = img_f.read()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    # 返回 JSON
    return jsonify({
        'log': detect_log,
        'image_base64': img_base64
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
