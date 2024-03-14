import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np

# 객체 탐지 모델 불러오기
def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 91  # COCO 데이터셋의 클래스 수
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# 이미지에서 객체 탐지 함수 정의
def detect_objects(image_path, threshold=0.5):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 이미지를 PyTorch 텐서로 변환
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)
    
    # 모델 불러오기
    model = load_model()
    
    # 모델을 evaluation 모드로 설정
    model.eval()
    
    # 이미지를 모델에 전달하여 객체 탐지 수행
    with torch.no_grad():
        prediction = model([image_tensor])
    
    # 탐지 결과를 이미지에 표시
    for box, score, label in zip(prediction[0]['boxes'], prediction[0]['scores'], prediction[0]['labels']):
        if score >= threshold:
            box = [int(coord) for coord in box]
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(image, f'Label: {label.item()} - Score: {score.item():.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image

# 객체 탐지 수행
image_path = 'example.jpg'  # 이미지 파일 경로 지정
result_image = detect_objects(image_path)

# 결과 이미지 표시
cv2.imshow('Object Detection Result', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
