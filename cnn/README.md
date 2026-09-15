# CNN 정리

# CNN 모델

class CatDogCNN(nn.Module): #catdogcnn이라는 모델을 만들고 파이토치의 신경망 기능을 사용할 수 있게 한다.

    def __init__(self): #객체가 만들어질때 자동으로 실행되는 함수
        super().__init__() #파이토치 신경망 모델로 제대로 사용할 수 있도록 기본 설정

        # RGB 이미지를 받아 16개의 특징 추출
        self.conv1 = nn.Conv2d( #2차원 이미지에서 합성곱을 적용하는 파이토치 함수(선, 모서리, 색깔 등 찾아냄)
            3, 16, kernel_size=3, padding=1 # rgb라서 3개, 이미지를 분석해서 16개의 특징, 특징 찾을때 사용하는 필터 크기,가장자리에 1칸 추가
        )

        # 16개의 특징을 받아 32개의 특징 추출
        self.conv2 = nn.Conv2d( #2차원 이미지에서 합성곱을 적용하는 파이토치 함수(선, 모서리, 색깔 등 찾아냄)
            16, 32, kernel_size=3, padding=1  #  이미지를 분석해서 32개의 특징, 특징 찾을때 사용하는 필터 크기,가장자리에 1칸 추가
        )

        # 32개의 특징을 받아 64개의 특징 추출
        self.conv3 = nn.Conv2d( #2차원 이미지에서 합성곱을 적용하는 파이토치 함수(선, 모서리, 색깔 등 찾아냄)
            32, 64, kernel_size=3, padding=1 #이미지를 분석해서 64개의 특징, 특징 찾을때 사용하는 필터 크기,가장자리에 1칸 추가
        )

        # 활성화 함수
        self.relu = nn.ReLU() #음수는 0으로, 양수는 그대로 바꾸는 활성화 함수

        # 이미지 크기를 절반으로 줄임
        self.pool = nn.MaxPool2d(
            kernel_size=2, #가장 큰 값 선택
            stride=2 #2칸씩 이동 -> 크기 절반 정도로 줄어든다.
        )

        # 특징을 128개의 값으로 연결
        self.fc1 = nn.Linear( #완전연결층(최종적 분류)
            64 * 16 * 16,
            128 # 64×16×16개의 특징을 입력받아 128개의 값으로 변환
        )

        # 고양이와 강아지 2가지로 분류
        self.fc2 = nn.Linear(
            128,
            2 #128개의 값을 받아서 2개의 값으로 만든다.(강아지 고양이)
        )

    def forward(self, x): #입력 데이터가 실제로 어떻게 통과할지 정하는 함수

        # CNN → ReLU → Pooling
        x = self.pool(self.relu(self.conv1(x))) #특징 찾기 -> relu 적용 -> 크기 절반으로 감소

        # CNN → ReLU → Pooling
        x = self.pool(self.relu(self.conv2(x))) #특징 찾기 -> relu 적용 -> 크기 절반으로 감소

        # CNN → ReLU → Pooling
        x = self.pool(self.relu(self.conv3(x))) #특징 찾기 -> relu 적용 -> 크기 절반으로 감소

        # 특징을 1차원으로 펼침
        x = torch.flatten(x, 1) #첫번째 차원만 그대로 두고, 나머지는 일렬로 펼침

        # 완전연결층
        x = self.relu(self.fc1(x)) #완전 연결층을 통과하고 활성화 함수 적용

        # 고양이 또는 강아지로 분류
        x = self.fc2(x) #출력

        return x #최종 결과 반환

# GPU가 있으면 GPU 없으면 CPU 사용

device = torch.device(
"cuda" if torch.cuda.is_available() else "cpu"
)

model = CatDogCNN().to(device) #만든 cnn모델을 실제 생성하고 gpu나 cpu로 이동

# 손실 함수와 최적화 함수

criterion = nn.CrossEntropyLoss() #손실함수 (틀리면 손실이 커지고, 맞히면 손실이 작아진다.)

optimizer = optim.Adam( #모델의 가중치를 업데이트해서 학습시키는 방법.
model.parameters(), #무엇을 학습시킬지 알려준다.
lr=0.001 #학습률 값

)
