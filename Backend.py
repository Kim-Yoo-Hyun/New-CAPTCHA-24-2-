from flask import Flask, jsonify, request, send_file
import numpy as np
import json
import os
from PIL import Image, ImageDraw
import cv2
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)

GENERATED_IMAGES_PATH = os.path.abspath(r'C:\Python312\CAPTCHA\generated_images')
FAKE_ANSWER_PATH = os.path.abspath(r'C:\Python312\CAPTCHA\fake_answer')
CORRECT_ANSWER_PATH = os.path.abspath(r'C:\Python312\CAPTCHA\correct_answer')
CAPTCHA_PATH = os.path.abspath(r'C:\Python312\CAPTCHA\static\generated_captcha.png')  # .png 확장자 추가

selected_method_global = None  # 글로벌 변수 추가

def generate_captcha_and_answer():
    global selected_method_global  # 글로벌 변수 사용
    try:
        selected_method = random.choice(['DCGANs', 'Diffusion'])
        selected_method_global = selected_method  # 선택된 메서드 저장

        # CAPTCHA 이미지 파일을 읽어서 무작위 선택
        generated_images_path = os.path.join(GENERATED_IMAGES_PATH, selected_method)
        image_files = [os.path.join(root, filename) 
                        for root, _, files in os.walk(generated_images_path)
                        for filename in files if filename.endswith('png')] 
        if not image_files:
            raise FileNotFoundError("No CAPTCHA images data found in the specified directories.")

        # 무작위로 선택된 CAPTCHA 이미지 저장
        chosen_top = random.choice(image_files)
        captcha_img = Image.open(chosen_top)
        captcha_img.save(CAPTCHA_PATH, format='PNG') 

        chosen_dir_name = os.path.basename(os.path.dirname(chosen_top))
        correct_dir_path = os.path.join(CORRECT_ANSWER_PATH, selected_method, chosen_dir_name)
        fake_dir_path = os.path.join(FAKE_ANSWER_PATH, selected_method, chosen_dir_name)
        
        if not os.path.exists(correct_dir_path) or not os.path.exists(fake_dir_path):
            raise FileNotFoundError(f"No matching directory.")

        # 정답 이미지 선택 (3~5개)
        correct_images = [os.path.join(correct_dir_path, filename)
                           for filename in os.listdir(correct_dir_path) if filename.endswith('.png')]
        # if len(correct_images) < 3:
        #    raise FileNotFoundError("Not enough correct images.")
        correct_count = random.randint(3, min(6, len(correct_images)))
        selected_correct = random.sample(correct_images, correct_count)

        # 오답 이미지 선택 (4~6개)
        fake_images = [os.path.join(fake_dir_path, filename)
                        for filename in os.listdir(fake_dir_path) if filename.endswith('.png')]
        #if len(fake_images) < 6:
        #    raise FileNotFoundError("Not enough fake images.")
        fake_count = 12 - correct_count
        selected_fake = random.sample(fake_images, fake_count)

        # 정답과 오답 이미지를 섞음
        combined_images = selected_fake + selected_correct
        random.shuffle(combined_images)

        # 기존 이미지 제거 후 새로 저장
        static_dir = r'C:\Python312\CAPTCHA\static'
        for file in os.listdir(static_dir):
            if file.startswith("answer_captcha"):
                os.remove(os.path.join(static_dir, file))

        # ANSWER_PATH에 각기 다른 이미지 경로 저장
        for idx, img_path in enumerate(combined_images):
            filename = f"answer_captcha_{idx + 1}_"
            if img_path in selected_correct:
                filename += "correct.png"
            else:
                filename += "fake.png"
            Image.open(img_path).save(os.path.join(static_dir, filename), format='PNG')

    except Exception as e:
        print(f"Error in generate_captcha_and_answer: {e}")


@app.route('/generate_captcha_metadata', methods=['GET'])
def generate_captcha_metadata():
    try:
        if selected_method_global is None:
            raise ValueError("CAPTCHA has not been generated yet.")
        # 선택된 메서드 반환
        return jsonify({'method': selected_method_global})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 상단 CAPTCHA 이미지 반환
@app.route('/generated_captcha.png', methods=['GET'])
def get_captcha_question():
    return send_file(CAPTCHA_PATH, mimetype='image/png')

@app.route('/get_captcha_files', methods=['GET'])
def get_captcha_files():
    try:
        files = []
        # 1~13번 이미지에 대해 'answer_captcha'로 시작하는 correct, fake 파일만 찾아 목록에 추가
        for i in range(1, 13):
            for status in ['correct', 'fake']:
                file_name = f'answer_captcha_{i}_{status}.png'
                static_dir = r'C:\Python312\CAPTCHA\static'
                if file_name.startswith('answer_captcha'):  # 'answer_captcha'로 시작하는 파일만 필터링
                    file_path = os.path.join(static_dir, file_name)
                    if os.path.exists(file_path):
                        files.append(file_name)
        return jsonify(files)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 하단 CAPTCHA 이미지 반환
@app.route('/answer_captcha_<int:index>_<status>.png', methods=['GET'])
def get_captcha_answer(index, status):
    try:
        if 1 <= index <= 12 and status in ["correct", "fake"]:
            static_dir = r'C:\Python312\CAPTCHA\static'
            
            file_name = f'answer_captcha_{index}_{status}.png'
            file_path = os.path.join(static_dir, file_name)

            if os.path.exists(file_path):
                return send_file(file_path, mimetype='image/png')
            return jsonify({'error': f'File not found: {file_name}'}), 404
        else:
            return jsonify({'error': 'Invalid index'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 제출된 경로 평가
@app.route('/submit', methods=['POST'])
def submit_path():
    """사용자가 선택한 정답 이미지 검증."""
    try:
        data = request.json
        name = data.get('name', '')
        dob = data.get('dob', '')
        username = data.get('username', '')
        password = data.get('password', '')
        email = data.get('email', '')
        selected_images = data.get('selected_images', [])

        if not selected_images:
            return jsonify({'message': 'No images selected.', 'success': False}), 400
        
        static_dir = r'C:\Python312\CAPTCHA\static'

        # 정답 이미지 파일 목록 생성: 'correct.png'로 끝나는 파일만 선택
        correct_image_names = [
            os.path.basename(filename) for filename in os.listdir(static_dir)
            if filename.endswith('correct.png')
        ]
        
        # 선택한 이미지가 정답인지 확인
        is_correct = (all(image_name in correct_image_names for image_name in selected_images) and 
                      len(selected_images) == len(correct_image_names))

        # 정답인지 오답인지 확인
        if is_correct:
            #hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            # 정답일 경우 개인정보 저장
            user_data = {
                'name': name,
                'dob': dob,
                'username': username,
                'original_password': password,  # 실제로는 암호화해야 함 (예: bcrypt 사용)
                #'hashed_password': hashed_password.decode('utf-8'),
                'email': email,
                'selected_images': selected_images
            }

            user_data_dir = os.path.abspath(r'C:\Python312\CAPTCHA\user_data')
            os.makedirs(user_data_dir, exist_ok=True)
            user_file_path = os.path.join(user_data_dir, f"{name}.json")

            with open(user_file_path, 'w') as user_file:
                json.dump(user_data, user_file, indent=4)

            generate_captcha_and_answer()
            # 정답일 경우
            return jsonify({'message': '가입이 완료되었습니다!', 'success': True})
        else:
            # 오답일 경우 새로운 CAPTCHA 생성
            generate_captcha_and_answer()
            return jsonify({'message': '사람이 아닙니다!', 'success': False}), 400
            
    except Exception as e:
        return jsonify({'message': '오류입니다'}), 500
        # return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    generate_captcha_and_answer()
    app.run(debug=True, port=5000)