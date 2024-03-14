import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models

# 데이터 증강 및 전처리
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 랜덤 리사이즈 및 잘라내기
    transforms.RandomHorizontalFlip(),  # 랜덤으로 이미지 좌우 반전
    transforms.ToTensor(),  # 이미지를 Tensor로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 이미지 정규화
])

# CIFAR-100 데이터셋 다운로드 및 로드
batch_size = 64
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ResNet-101 모델 정의
model = models.resnet101(weights=None)  # 사전 학습된 가중치를 사용하지 않습니다.

# 분류기를 100개의 클래스에 맞게 변경
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 100)

# GPU 사용 가능 여부 확인 및 모델을 GPU로 이동
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 손실 함수, 옵티마이저 및 학습률 스케줄러 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 5 에폭마다 학습률을 0.1씩 감소

# 학습 함수 정의
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

# 검증 함수 정의
def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# 학습 및 검증 실행
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, test_loader, criterion, device)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    scheduler.step()  # 학습률 스케줄러 업데이트

# 학습이 완료된 후에 모델 저장
torch.save(model.state_dict(), 'resnet101_cifar100.pth')

# 모델 로드 및 테스트 데이터셋에 대한 정확도 출력
model = models.resnet101(weights=None)
model.fc = nn.Linear(model.fc.in_features, 100)  # 모델의 분류기 변경
model.load_state_dict(torch.load('resnet101_cifar100.pth'))
model = model.to(device)

test_loss, test_acc = validate(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
