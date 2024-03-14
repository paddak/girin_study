# etc
import os, sys
import glob
import csv
import cv2
import tqdm
import random
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# torch library
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# torchvision library
import torchvision
from torchvision import transforms, models
import torch.optim as optim

# 재현성을 위한 랜덤시드 고정
random_seed = 2023
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

# 데이터 전처리
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        # transforms.RandomChoice([
        # transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomAffine(
        #     degrees=15, translate=(0.2, 0.2),
        #     scale=(0.8, 1.2), shear=15, resample=Image.BILINEAR)
    # ]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])             

# 배치 사이즈와 train:validation 비율 정의
batch_size = 256
val_size = 0.2

# torchvision에서 제공하는 CIFAR10 학습 데이터셋 다운로드
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
val_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=val_transform)

# Train 데이터에서 일정 비율 vaildation data 분리
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(val_size * num_train))
train_idx, val_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

# 데이터로더 정의
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          sampler=train_sampler, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                          sampler=val_sampler, num_workers=2)

# torchvision에서 제공하는 CIFAR10 테스트 데이터셋 다운로드
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=val_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# 클래스 정의
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 이미지 데이터 시각화
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 학습 이미지 얻기
dataiter = iter(train_loader)
images, labels = next(dataiter)
# 이미지 출력
imshow(torchvision.utils.make_grid(images))
# 라벨 프린트
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# # 테스트를 위한 Custom Dataset 다운로드
# !gdown https://drive.google.com/uc?id=1GTES_wxB8b-jsZIqHgNyV9pEgpMLtfzc
# # 파일 압축 해제
# !unzip ./03_classification_custom_dataset.zip

# 커스텀 데이터셋 클래스
class CUSTOMDataset(Dataset):
    def __init__(self, mode: str = 'test', transforms: transforms = None):
        self.mode = mode
        self.transforms = transforms
        self.images = []
        self.labels = []

        for folder in os.listdir('./custom_dataset'):
          files = os.path.join('./custom_dataset', folder)
          if folder == '.DS_Store':
            continue
          files_path = os.listdir(files)
          for file in files_path:
            self.images.append(os.path.join(files,file))
            self.labels.append(classes.index(folder))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image = Image.open(self.images[index]).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        image = np.array(image)
        label = self.labels[index]
        return image, label
    
# 커스텀 데이터셋 & 로더
custom_dataset = CUSTOMDataset('test', transforms = val_transform)
custom_loader = DataLoader(
    custom_dataset, batch_size=16, shuffle=False, num_workers=2)

# 디바이스 체크 & 할당
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
model = model.to(device)

#loss, optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train(epoch):
  train_loss = 0.0
  model.train()
  for i, data in enumerate(tqdm.tqdm(train_loader), 0):
      # 입력 데이터 가져오기 data: [inputs, labels]
      inputs, labels = data[0].to(device), data[1].to(device)

      # parameter gradients를 제로화
      optimizer.zero_grad()

      # 입력 이미지에 대한 출력 생성
      outputs = model(inputs)

      # 손실함수 계산 밎 업데이트
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

  return train_loss

def val():
  val_loss = 0.0
  val_accuracy = 0.0
  with torch.no_grad():
    # 모델 평가 모드 설정
    model.eval()
    for i, data in enumerate(tqdm.tqdm(val_loader), 0):
      # 입력 데이터 가져오기 data: [inputs, labels]
      inputs, labels = data[0].to(device), data[1].to(device)

      # 입력 이미지에 대한 출력 생성
      outputs = model(inputs)

      # 손실함수 계산
      loss = criterion(outputs, labels)
      val_loss += loss.item()

      # 예측 라벨
      _, predicted = torch.max(outputs, 1)

      # accuracy 계산
      val_accuracy += (predicted == labels).sum().item()

  return val_loss, val_accuracy

def test(test_loader):
    correct = 0
    total = 0
    correct_class = {classname: 0 for classname in classes}
    total_class = {classname: 0 for classname in classes}
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # 입력 이미지에 대한 출력 생성
            outputs = model(inputs)
            # 예측 라벨
            _, predicted = torch.max(outputs.data, 1)
            # 전체 정확도 계산
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # 클래스 별 정확도 계산
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_class[classes[label]] += 1
                total_class[classes[label]] += 1
    # 전체 정확도 출력
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    # 클래스 별 정확도 출력
    for classname, correct_count in correct_class.items():
        if total_class[classname] == 0:
          continue
        accuracy = 100 * float(correct_count) / total_class[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# 학습 에포크 설정
train_epochs = 20
best_acc = 0.0
# 모델 저장 경로 정의
model_path = './cifar_resnet.pth'
for epoch in range(train_epochs):
  # 학습 메소드 실행
  train_loss = train(epoch)
  print(f'[{epoch + 1}] loss: {train_loss / len(train_loader):.3f}')
  # 검증 메소드 실행
  val_loss, val_acc = val()
  vaild_acc = val_acc / (len(val_loader)*batch_size)
  print(f'[{epoch + 1}] loss: {val_loss / len(val_loader):.3f} acc: {vaild_acc:.3f}')
  # 정확도가 기존 베스트를 갱신할 경우 모델 저장
  if vaild_acc >= best_acc:
    best_acc = vaild_acc
    torch.save(model.state_dict(), model_path)
print('Finished Training')


#평가
model_path = '/content/cifar_resnet.pth'
# 모델 가중치 로드
model.load_state_dict(torch.load(model_path))
# 테스트 메소드 실행
test(custom_loader)
model_path = '/content/cifar_resnet.pth'
# 모델 가중치 로드
model.load_state_dict(torch.load(model_path))
# 테스트 메소드 실행
test(test_loader)