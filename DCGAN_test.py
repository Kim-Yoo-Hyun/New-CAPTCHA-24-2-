import os
import json
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
import numpy as np

# GPU 설정
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
print(f"Using device: {DEVICE}")

# 데이터셋 경로
DATASET_PATH = r"C:\Users\morri\Downloads\archive"
OUTPUT_PATH = "generated_images"
OUTPUT_JSON = "output_path_data"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(OUTPUT_JSON, exist_ok=True)

classes = ['truck', 'train', 'seaplane', 'motorcycle', 'motorbus', 'boat', 'bicycle', 'airplane']

label_map = {
    'motorcycle': 0, 'seaplane': 1, 'boat': 2, 'motorbus': 3,
    'bicycle': 4, 'train': 5, 'truck': 6, 'airplane': 7
}

# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for subdir, _, files in os.walk(self.root):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    class_name = os.path.basename(subdir)
                    if class_name in label_map:
                        self.image_paths.append(os.path.join(subdir, file))
                        self.labels.append(label_map[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# 이미지 변환
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 데이터 로더 생성
dataset = CustomDataset(DATASET_PATH, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
noise_size = 100

# Generator 모델 정의
class Generator(nn.Module):
    def __init__(self, noise_size, num_classes):
        super(Generator, self).__init__()
        self.noise_size = noise_size
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.gan = nn.Sequential(
            nn.ConvTranspose2d(noise_size + num_classes, 1024, kernel_size=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
             nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # 라벨 임베딩 후 병합
        label_embeddings = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        label_embeddings = label_embeddings.expand(-1, -1, noise.size(2), noise.size(3))
        input = torch.cat((noise, label_embeddings), dim=1)
        return self.gan(input) 

# Discriminator 모델 정의
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes

        # 레이블 임베딩
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.disc = nn.Sequential(
            nn.Conv2d(3 + num_classes, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=4),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # 레이블 임베딩
        label_embeddings = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        label_embeddings = label_embeddings.expand(-1, -1, img.size(2), img.size(3))
        input = torch.cat((img, label_embeddings), dim=1)
        return self.disc(input)

# 모델 초기화
num_classes = len(classes)
generator = Generator(noise_size, num_classes).to(DEVICE)
discriminator = Discriminator(num_classes).to(DEVICE)

# 초기 가중치 설정
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

generator.apply(weights_init)
discriminator.apply(weights_init)

# 손실 함수 및 최적화 설정
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 테두리 추출 및 JSON 저장 함수
def extract_and_save_contours(image, save_path):
    # grayscale로 이미지 전환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = {
            'top_left': {'x': x / image.shape[1], 'y': y / image.shape[0]},
            'top_right': {'x': (x + w) / image.shape[1], 'y': y / image.shape[0]},
            'bottom_left': {'x': x / image.shape[1], 'y': (y+h) / image.shape[0]},
            'bottom_right': {'x': (x + w) / image.shape[1], 'y': (y + h) / image.shape[0]}
        }
        bounding_boxes.append(box)
    # contour 데이터 준비 
    #contour_data = []
    #for contour in contours:
    #    points = [{'x': point[0][0] / image.shape[1], 'y': point[0][1] / image.shape[0]} for point in contour]
    #    contour_data.append({'points': points})

    with open(save_path, 'w') as f:
        json.dump(bounding_boxes, f, indent=4)

# 학습 루프
num_epochs = 100
for epoch in range(num_epochs):
    for i, (real_imgs, labels) in enumerate(data_loader):
        real_imgs = real_imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        batch_size = real_imgs.size(0)

        # 라벨 설정
        valid = torch.ones((batch_size, 1, 1, 1), device=DEVICE)
        fake = torch.zeros((batch_size, 1, 1, 1), device=DEVICE)

        # Discriminator 학습
        optimizer_D.zero_grad()

        # 진짜 이미지에 대한 손실
        real_loss = criterion(discriminator(real_imgs, labels), valid)

        # 가짜 이미지에 대한 손실
        z = torch.randn(batch_size, noise_size, 1, 1, device=DEVICE)
        fake_labels = torch.randint(0, num_classes, (batch_size,), device=DEVICE)
        fake_imgs = generator(z, fake_labels)
        fake_loss = criterion(discriminator(fake_imgs.detach(), fake_labels), fake)

        # Dicriminator 총 손실
        loss_D = (real_loss + fake_loss) / 2
        loss_D.backward()
        optimizer_D.step()

        # Generator 학습
        optimizer_G.zero_grad()

        # Generator 손실 (Discriminator가 진짜로 판단하도록 학습)
        loss_G = criterion(discriminator(fake_imgs, fake_labels), valid)
        loss_G.backward()
        optimizer_G.step()

        # 진행 상태 출력
        if i % 64 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(data_loader)}] "
                  f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

generator.eval()
with torch.no_grad():
    for class_name in classes:
        class_folder = os.path.join(OUTPUT_PATH, class_name)
        os.makedirs(class_folder, exist_ok=True)

        class_idx = classes.index(class_name)
        z = torch.randn(16, noise_size, 1, 1, device=DEVICE)
        labels = torch.full((16,), class_idx, dtype=torch.long, device=DEVICE)
        gen_imgs = generator(z, labels)
    
        # for class_name in classes:
        #   os.makedirs(os.path.join(OUTPUT_PATH, class_name), exist_ok=True)
        #   os.makedirs(os.path.join(OUTPUT_JSON, class_name), exist_ok=True)

        for idx, img in enumerate(gen_imgs):
            # class_name = classes[idx % len(classes)]
            # img_folder = os.path.join(OUTPUT_PATH, class_name)
            img_path = os.path.join(class_folder, f"final_img_{idx}.png")
            save_image(0.5 * img + 0.5, img_path)
        
            json_folder = os.path.join(OUTPUT_JSON, class_name)
            os.makedirs(json_folder, exist_ok=True)
            json_path = os.path.join(json_folder, f"final_img_{idx}.json")

            # OpenCV로 이미지를 읽고 테두리 저장
            opencv_img = cv2.imread(img_path)
            extract_and_save_contours(opencv_img, json_path)
