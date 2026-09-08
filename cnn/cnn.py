import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 이미지 크기를 128x128로 통일하고 Tensor로 변환
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# 학습 데이터와 테스트 데이터 불러오기
train_data = datasets.ImageFolder(
    "./data/train",
    transform=transform
)

test_data = datasets.ImageFolder(
    "./data/test",
    transform=transform
)


# 데이터를 32개씩 묶어서 사용
train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False
)

print("분류 종류:", train_data.classes)


# CNN 모델
class CatDogCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # RGB 이미지를 받아 16개의 특징 추출
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, padding=1
        )

        # 16개의 특징을 받아 32개의 특징 추출
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, padding=1
        )

        # 32개의 특징을 받아 64개의 특징 추출
        self.conv3 = nn.Conv2d(
            32, 64, kernel_size=3, padding=1
        )

        # 활성화 함수
        self.relu = nn.ReLU()

        # 이미지 크기를 절반으로 줄임
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 특징을 128개의 값으로 연결
        self.fc1 = nn.Linear(
            64 * 16 * 16,
            128
        )

        # 고양이와 강아지 2가지로 분류
        self.fc2 = nn.Linear(
            128,
            2
        )

    def forward(self, x):

        # CNN → ReLU → Pooling 
        x = self.pool(self.relu(self.conv1(x)))

        # CNN → ReLU → Pooling
        x = self.pool(self.relu(self.conv2(x)))

        # CNN → ReLU → Pooling
        x = self.pool(self.relu(self.conv3(x)))

        # 특징을 1차원으로 펼침
        x = torch.flatten(x, 1)

        # 완전연결층
        x = self.relu(self.fc1(x))

        # 고양이 또는 강아지로 분류
        x = self.fc2(x)

        return x


# GPU가 있으면 GPU 없으면 CPU 사용
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CatDogCNN().to(device)


# 손실 함수와 최적화 함수
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# 5번 반복해서 학습
epochs = 5

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # 이전 기울기 초기화
        optimizer.zero_grad()

        # 이미지 예측
        outputs = model(images)

        # 예측값과 정답 비교
        loss = criterion(outputs, labels)

        # 오차를 이용해 학습
        loss.backward()

        # 가중치 업데이트
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {total_loss:.4f}"
    )


# 테스트
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        # 테스트 이미지 예측
        outputs = model(images)

        # 가장 높은 값을 가진 결과 선택
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


# 정확도 계산
accuracy = 100 * correct / total

print(f"테스트 정확도: {accuracy:.2f}%")


# 학습한 모델 저장
torch.save(
    model.state_dict(),
    "cat_dog_cnn.pth"
)

print("모델 저장 완료!")