import os
import json
import torch
import numpy as np
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from keras import layers, models

# 경로 정의 (이미지 및 JSON 파일 저장 위치)
OUTPUT_PATH = r"C:\Python312\CAPTCHA\generated_images"  # GAN 생성 이미지 저장 경로
OUTPUT_JSON = r"C:\Python312\CAPTCHA\output_path_data"  # JSON 파일 저장 경로

# CNN 모델 생성 함수
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(64 * 64, activation='sigmoid'),
        layers.Reshape((64, 64, 1))
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model

# 사용자 마우스 경로로 히트맵 생성 함수
def create_heatmap(mouse_data, width=64, height=64):
    heatmap = np.zeros((height, width), dtype=np.float32)
    for contour in mouse_data:
        for point in contour['points']:
            x, y = int(point['x'] * width), int(point['y'] * height)
            if 0 <= x < width and 0 <= y < height:
                heatmap[y, x] += 1
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)

    max_val = heatmap.max()
    if max_val > 0:
        heatmap = (heatmap / max_val) * 255
    else:
        heatmap.fill(0)
    return heatmap.astype(np.uint8)

# 사용자 데이터셋 클래스 정의
class CustomHeatmapDataset(Dataset):
    def __init__(self, image_path, json_path, transform=None):
        self.image_files = []
        self.json_files = []
        
        for class_folder in os.listdir(image_path):
            class_image_dir = os.path.join(image_path, class_folder)
            class_json_dir = os.path.join(json_path, class_folder)
            if os.path.isdir(class_image_dir) and os.path.isdir(class_json_dir):
                for file in os.listdir(class_image_dir):
                    if file.endswith(('png', 'jpg', '.jpeg')):
                        self.image_files.append(os.path.join(class_image_dir, file))
                        self.json_files.append(os.path.join(class_json_dir, file.replace('.png', '.json')
                                                                        .replace('.jpg', 'json')
                                                                        .replace('.jpeg', '.json')))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        json_file = self.json_files[idx]

        # image = cv2.imread(os.path.join(self.image_path, image_file))
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (64, 64))
        if self.transform:
            image = self.transform(image)

        # json_file = image_file.replace('.png', '.json').replace('.jpg', '.json').replace('.jpeg', '.json')
        with open(json_file, 'r') as f:
            mouse_data = json.load(f)

        heatmap = create_heatmap(mouse_data)
        return image, heatmap

# 이미지 변환 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 데이터 로더 생성
dataset = CustomHeatmapDataset(OUTPUT_PATH, OUTPUT_JSON, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 모델 학습을 위한 데이터 준비
x_train = []
y_train = []

for images, heatmaps in data_loader:
    # PyTorch 텐서를 NumPy 배열로 변환
    images = images.numpy().transpose(0, 2, 3, 1)  # (batch_size, channels, height, width) -> (batch_size, height, width, channels)
    heatmaps = np.expand_dims(heatmaps.numpy(), axis=1)
    
    # 채널 차원을 추가하여 (batch_size, height, width, 1) 형식으로 변환

    x_train.append(images)
    y_train.append(heatmaps)

# 데이터 배열을 하나의 큰 배열로 변환
x_train = np.concatenate(x_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

# CNN 모델 초기화 및 학습
cnn_model = create_cnn_model()
cnn_model.fit(x_train, y_train, epochs=10, batch_size=32)

# 학습된 모델 저장
cnn_model.save('path_similarity_model.h5')
print("모델이 'path_similarity_model.h5'로 저장되었습니다.")
