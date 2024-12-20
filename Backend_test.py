from flask import Flask, jsonify, request, send_file
from tensorflow.keras.models import load_model
import numpy as np
import json
import os
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)

# OUTPUT_DIR = "C:\Python312\CAPTCHA"
GENERATED_IMAGES_PATH = os.path.abspath(r'C:\Python312\CAPTCHA\generated_images')
OUTPUT_JSON_PATH = os.path.abspath(r'C:\Python312\CAPTCHA\output_path_data')
# STATIC_DIR = "static"
HEATMAP_PATH = os.path.abspath(r'C:\Python312\CAPTCHA\static\heatmap.png')
CAPTCHA_PATH = os.path.abspath(r'C:\Python312\CAPTCHA\static\generated_captcha.png')  # .png 확장자 추가
PATH_DATA_PATH = os.path.abspath(r'C:\Python312\CAPTCHA\static\path_data.json')  # 예시 파일 경로

# 모델 로드
cnn_model = load_model("path_similarity_model.h5")

def generate_captcha_and_path():
    try:
        image_files = []
        json_files = []    

        # 서브폴더 포함 파일 탐색
        for root, _, files in os.walk(GENERATED_IMAGES_PATH):
            for filename in files:
                if filename.endswith('.png'):
                    image_files.append(os.path.join(root, filename))

        for root, _, files in os.walk(OUTPUT_JSON_PATH):
            for filename in files:
                if filename.endswith('.json'):
                    json_files.append(os.path.join(root, filename))

        if not image_files or not json_files:
            raise FileNotFoundError("No images or path data found in the specified directories.")

        chosen_image = random.choice(image_files)
        chosen_json = chosen_image.replace('.png', '.json').replace(GENERATED_IMAGES_PATH, OUTPUT_JSON_PATH)

        captcha_path = chosen_image
        json_path = chosen_json

        captcha_img = Image.open(captcha_path)
        captcha_img.save(CAPTCHA_PATH, format='PNG')

        with open(json_path, 'r') as f:
            correct_path = json.load(f)
            with open(PATH_DATA_PATH, 'w') as path_file:
                json.dump(correct_path, path_file)
        
    except Exception as e:
        print(f"Error in generate_captcha_and_path: {e}")
        raise

def create_heatmap(mouse_data, width=64, height=64):
    heatmap = np.zeros((height, width), dtype=np.float32)
    for point in mouse_data:
        x, y = int(point['x'] * width), int(point['y'] * height)
        if 0 <= x < width and 0 <= y < height:
            heatmap[y, x] += 1
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max()) * 255
    else:
        heatmap = np.zeros_like(heatmap)
    
    # heatmap = np.nan_to_num(heatmap, nan=0.0)
    # heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
    return heatmap.astype(np.uint8)

def calculate_similarity(user_path, correct_path):
    user_points = np.array([[p['x'], p['y']] for p in user_path])  # 사용자 경로 좌표
    correct_points = np.array([[p['x'], p['y']] for p in correct_path])  # 정답 경로 좌표
    distances = np.sqrt(np.sum((user_points - correct_points) ** 2, axis=1))  # 거리 계산
    return float(np.exp(-distances.mean()))  # 평균 거리에 기반한 유사도 반환

# CAPTCHA 이미지 반환
@app.route('/generated_captcha.png', methods=['GET'])
def get_captcha():
    # generate_captcha_and_path()
    return send_file(CAPTCHA_PATH, mimetype='image/png')

# 경로 데이터 반환
@app.route('/path_data.json', methods=['GET'])
def get_path_data():
    with open(PATH_DATA_PATH, 'r') as f:
        data = json.load(f)
    return jsonify(data)

# 제출된 경로 평가
@app.route('/submit', methods=['POST'])
def submit_path():
    try:
        data = request.json
        user_path = data['mouseData', []]
        name = data.get('name', '')
        dob = data.get('dob', '')
        username = data.get('username', '')
        password = data.get('password', '')
        email = data.get('email', '')

        if not user_path:
            raise ValueError("Mouse data is missing")
        
        generate_captcha_and_path()

        # Heatmap 생성
        heatmap = create_heatmap(user_path)
        cv2.imwrite(HEATMAP_PATH, heatmap)
        
        # 정답 경로 로드
        with open(PATH_DATA_PATH, 'r') as f:
            correct_path = json.load(f)

        # CNN 모델 예측
        heatmap_input = heatmap.reshape(1, 64, 64, 1) / 255.0
        score = cnn_model.predict(heatmap_input)[0][0]

        # 경로 유사도 계산
        similarity = calculate_similarity(user_path, correct_path)

        response_message = "사용자 확인" if similarity > 0.7 else "의심스러운 사용자"
        
        user_data = {
            'name': name,
            'dob': dob,
            'username': username,
            'password': password,  # 실제로는 암호화해야 함 (예: bcrypt 사용)
            'email': email,
            'mouseData': user_path,
            'similarity_score': similarity,
            'model_score': model_score
        }

        user_data_dir = os.path.abspath(r'C:\Python312\CAPTCHA\user_data')
        os.makedirs(user_data_dir, exist_ok=True)
        user_file_path = os.path.join(user_data_dir, f"{name}.json")

        with open(user_file_path, 'w') as user_file:
            json.dump(user_data, user_file, indent=4)
            
        return jsonify({'message': response_message, 'score': similarity, 'model_score': score})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    # os.makedirs(STATIC_DIR, exist_ok=True)
    app.run(debug=True, port=5000)