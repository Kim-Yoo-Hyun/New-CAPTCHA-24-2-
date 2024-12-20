import os
import random
import time
from pynput.mouse import Listener, Button
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Input
from keras.optimizers import Adam
from threading import Event

# 비행기 이미지 경로 설정
data_path = "C:\\Users\\morri\\Downloads\\airplane"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"경로를 찾을 수 없습니다: {data_path}")

# 마우스 좌표 저장
positions = []
stop_event = Event()  # 종료 이벤트
tracking = False
correct_path = []  # 올바른 경로를 저장할 리스트

# GAN 모델 생성 함수
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(64 * 64 * 3, activation='tanh'))
    model.add(Reshape((64, 64, 3)))
    return model

def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(64, 64, 3)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')
    return gan

def train_gan(model_path='C:/Python312/CAPTCHA/models/gan_model.h5', epochs=1000, batch_size=64):
    # 데이터셋 로드 및 전처리
    data_dir = "C:/Users/morri/Downloads/airplane"
    image_size = (64, 64)
    dataset = []
    for img_path in os.listdir(data_dir):
        img = cv2.imread(os.path.join(data_dir, img_path))
        if img is not None:
            img = cv2.resize(img, image_size)
            img = img / 255.0  # Normalize to [0, 1]
            dataset.append(img)
    dataset = np.array(dataset)

    # 모델 초기화
    discriminator = build_discriminator()
    generator = build_generator()
    discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

    gan = build_gan(generator, discriminator)
    gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

    # GAN 훈련
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # 진짜 이미지
        idx = np.random.randint(0, dataset.shape[0], batch_size)
        real_images = dataset[idx]

        # 가짜 이미지 생성
        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(noise)

        # 판별자 훈련
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # 생성자 훈련
        g_loss = gan.train_on_batch(noise, real_labels)

        # Epoch 결과 출력
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")

    # 모델 저장
    generator.save(model_path)
    print(f"모델이 {model_path}에 저장되었습니다.")

# GAN 모델을 불러와서 이미지 생성 및 경로 설정
def generate_captcha_with_path():
    model_path = 'C:/Python312/CAPTCHA/models/gan_model.h5'
    if not os.path.exists(model_path):
        print("GAN 모델이 존재하지 않습니다. 새로 생성합니다...")
        train_gan()

    model = load_model(model_path)

    # 비행기 데이터셋에서 랜덤 이미지를 선택
    airplane_images = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.png') or f.endswith('.jpg')]
    if not airplane_images:
        raise ValueError("비행기 이미지 데이터가 없습니다.")

    random_image_path = random.choice(airplane_images)
    airplane_image = cv2.imread(random_image_path)
    airplane_image = cv2.cvtColor(airplane_image, cv2.COLOR_BGR2RGB)

    # GAN으로 새로운 비행기 이미지 생성
    noise = np.random.randn(1, 100)  # 잠재 공간의 랜덤 노이즈
    generated_image = model.predict(noise)[0]  # 생성된 이미지

    # 이미지의 픽셀 범위 조정
    generated_image = (generated_image * 255).astype(np.uint8)
    generated_image = cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR)

    # 이미지 크기를 데이터셋 이미지 크기와 동일하게 조정
    generated_image = cv2.resize(generated_image, (airplane_image.shape[1], airplane_image.shape[0]))

    # 경로 생성 (비행기 테두리를 경로로 설정)
    global correct_path
    gray = cv2.cvtColor(airplane_image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        correct_path = [tuple(pt[0]) for pt in contours[0]]
    else:
        correct_path = []

    # 경로를 이미지에 표시
    for (x, y) in correct_path:
        cv2.circle(generated_image, (x, y), 2, (0, 255, 0), -1)

    # 생성 이미지 저장
    image_path = 'generated_airplane_with_path.png'
    cv2.imwrite(image_path, generated_image)

    # 경로 데이터 저장
    path_data_path = 'correct_path.txt'
    with open(path_data_path, 'w') as f:
        for x, y in correct_path:
            f.write(f"{x},{y}\n")

    print(f"생성된 이미지는 {image_path}에 저장되었습니다.")
    print(f"경로 데이터는 {path_data_path}에 저장되었습니다.")

    return generated_image, correct_path

# 마우스 클릭 분석
def on_click(x, y, button, pressed):
    global tracking
    if button == Button.left:
        if pressed:
            tracking = True
        else:
            tracking = False

# 마우스 움직임 분석
def on_move(x, y):
    global tracking
    if tracking:
        positions.append((x, y))

# 경로 검증 함수
def validate_path(positions, correct_path, threshold=50):
    if not correct_path:
        print("올바른 경로 데이터가 없습니다.")
        return False

    for i, (x, y) in enumerate(positions):
        if i >= len(correct_path):
            break
        cx, cy = correct_path[i]
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        if distance > threshold:
            return False
    return True

# 메인 함수
def main():
    global correct_path
    captcha_image, correct_path = generate_captcha_with_path()

    # 이미지 출력
    plt.imshow(cv2.cvtColor(captcha_image, cv2.COLOR_BGR2RGB))
    plt.title("Generated CAPTCHA with Path")
    plt.axis('off')
    plt.show()

    print("마우스 추적 시작. 클릭 후 이동하세요. 종료하려면 Ctrl+C를 누르세요.")
    listener = Listener(on_move=on_move, on_click=on_click)
    listener.start()

    try:
        while not stop_event.is_set():  # 종료 이벤트를 기다림
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n마우스 추적 종료. 히트맵 생성 중...")
        stop_event.set()
        listener.stop()

        # 데이터 정리
        if not positions:
            print("수집된 데이터가 없습니다.")
            return

        # 경로 검증
        if validate_path(positions, correct_path):
            print("올바른 경로를 따랐습니다.")
        else:
            print("잘못된 경로를 따랐습니다.")

        x_coords, y_coords = zip(*positions)

        # 화면 크기 확인 (데이터에 따라 수동 조정 가능)
        screen_width, screen_height = max(x_coords), max(y_coords)

        # 히트맵 데이터 준비
        heatmap, xedges, yedges = np.histogram2d(
            x_coords, y_coords, bins=(100, 100),
            range=[[0, screen_width], [0, screen_height]]
        )

        # 히트맵 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap.T,  # 히트맵 데이터
            cmap="Reds",  # 색상 맵을 더 부드럽게 설정
            cbar=True,
            xticklabels=False,
            yticklabels=False,
            linewidths=0.1,  # 각 셀 사이의 경계선 두께
            alpha=0.7,  # 히트맵의 투명도 조정
            square=False,  # 정사각형으로 표시하지 않음
            linecolor='black'  # 경계선 색상 설정
        )
        plt.title("Mouse Movement Heatmap")
        plt.xlabel("Screen X")
        plt.ylabel("Screen Y")
        plt.show()

if __name__ == "__main__":
    main()



