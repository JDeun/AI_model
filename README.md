# Image Classifier for Cat

## 프로젝트 개요

이 프로젝트는 딥러닝을 사용하여 고양이에게 유해한 식물의 이미지를 분류하는 시스템을 개발하고 사용하는 과정을 담고 있습니다. 프로젝트는 크게 두 부분으로 나뉩니다:
1. CNN 모델 훈련
2. 훈련된 모델을 사용한 이미지 분류

## 주요 기능

1. 데이터 전처리 및 증강
2. CNN 모델 설계 및 훈련
3. 모델 성능 평가 및 시각화
4. 훈련된 모델을 사용한 이미지 분류
5. 분류 결과에 따른 이미지 파일 정리

## 기술 스택

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Pillow (PIL)

## 설치 방법

1. 이 저장소를 클론합니다:
   ```
   git clone https://github.com/your-username/nepeta-cataria-classifier.git
   cd nepeta-cataria-classifier
   ```

2. 필요한 라이브러리를 설치합니다:
   ```
   pip install tensorflow numpy matplotlib pillow
   ```

## 프로젝트 구조

```
nepeta-cataria-classifier/
│
├── train_model.ipynb        # 모델 훈련 스크립트
├── image_classifier.py      # 이미지 분류 스크립트
├── models/                  # 훈련된 모델 저장 폴더
│   └── plant_classification_model.h5
├── dataset/                 # 훈련 데이터셋
│   ├── class_1/
│   └── class_2/
├── input_images/            # 분류할 이미지 폴더
└── output_images/           # 분류 결과 저장 폴더
    ├── can_use/
    └── can_not_use/
```

## 사용 방법

### 1. 모델 훈련

1. `train_model.ipynb` Jupyter 노트북을 엽니다.
2. 데이터 경로를 설정합니다:
   ```python
   data_dir = "path/to/your/dataset"
   ```
3. 필요에 따라 하이퍼파라미터(이미지 크기, 배치 크기, 에폭 등)를 조정합니다.
4. 노트북의 모든 셀을 실행합니다.

주요 코드 설명:
- 데이터 증강을 위해 `ImageDataGenerator`를 사용합니다.
- CNN 모델은 여러 개의 컨볼루션 레이어와 풀링 레이어로 구성됩니다.
- Nadam 옵티마이저와 categorical crossentropy 손실 함수를 사용합니다.
- 훈련 과정에서의 정확도와 손실을 시각화합니다.
- 훈련된 모델은 'plant_classification_model.h5' 파일로 저장됩니다.

### 2. 이미지 분류

1. 훈련된 모델 파일(.h5 형식)을 `models/` 폴더에 위치시킵니다.
2. 분류할 이미지들을 `input_images/` 폴더에 넣습니다.
3. 터미널에서 다음 명령을 실행합니다:
   ```
   python image_classifier.py
   ```
4. 프롬프트에 따라 모델 경로, 입력 이미지 폴더 경로, 출력 폴더 경로를 입력합니다.

## 모델 아키텍처

```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
```

## 성능 평가

모델의 성능은 validation set을 사용하여 평가됩니다. 훈련 과정에서 accuracy와 loss 그래프를 통해 모델의 성능을 시각적으로 확인할 수 있습니다.

## 주의사항

- 훈련 데이터셋은 적절히 레이블링되어 있어야 하며, 각 클래스별로 충분한 수의 이미지가 필요합니다.
- 이미지 분류 시 사용되는 이미지의 크기는 모델 훈련 시 사용된 크기(기본값: 150x150 픽셀)와 일치해야 합니다.
- GPU를 사용할 수 있는 환경에서 훈련을 진행하면 처리 속도가 크게 향상됩니다.

## 향후 개선 사항

- 더 많은 데이터로 모델 재훈련
- 다양한 모델 아키텍처 실험 (예: ResNet, VGG 등)
- 전이 학습(Transfer Learning) 적용
- 모델 해석 기능 추가 (예: Grad-CAM)

## 기여 방법

프로젝트 개선에 기여하고 싶으시다면:

1. 이 저장소를 포크합니다.
2. 새 브랜치를 만듭니다 (`git checkout -b feature/AmazingFeature`).
3. 변경 사항을 커밋합니다 (`git commit -m 'Add some AmazingFeature'`).
4. 브랜치에 푸시합니다 (`git push origin feature/AmazingFeature`).
5. Pull Request를 생성합니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 연락처

프로젝트 관리자 - gadi2003@naver.com
