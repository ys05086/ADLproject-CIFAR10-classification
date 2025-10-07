# AdvancedDeeplearningProject1
Advanced Deeplearning Project1 - Using CIFAR10 Dataset
고급 딥러닝 프로그래밍 수업 프로젝트 - CIFAR10 Dataset 학습

## Introduction
- 이 프로젝트는 CIFAR-10 예측에 있어서 높은 정확도를 목표로 함.
- 딥러닝 학습 파이프라인(데이터 전처리, 모델정의, 학습 루프 등을 포함함.

- CIFAR-10 데이터셋에는 10 class가 존재하며, 각각의 Class는 5천장의 학습데이터와, 1000장의 테스트데이터로 이루어져있음.

## Used Language / Framworks / Library
- Python 3.9
- PyTorch
- CUDA 12.8
- Numpy
- OpenCV
- Os
- tqdm

## Details & Features
- 데이터 전처리(Data preprocessing): 평균/표준편차를 이용한 정규화
- 데이터 증강(Data Augmentation): CutOut, Flip
- Residual Block(Pre-Activation) 기반 네트워크 구현
  - VGG-16 및 RESNET-18, RESNET-50을 구현함.
  - 또한, Pre-Activation RESNET을 구현함.
  - RoR(Residual Network of Residual Network) - 3 (Pre-Activation ver)을 구현함.(110)
- Checkpoint save, load

### 성능
- CIFAR-10 Test Accuracy: 91% (100k iterations, minibatch  size: 128) (Using RoR-3 110 model)
