# ADL Project - CIFAR-10 Classification
Advanced Deeplearning Project1 - Using CIFAR10 Dataset


## Introduction
- The goal of this project is to achieve high accuracy on CIFAR-10 image classification.
- This project implements a full deep learning pipeline, including data preprocessing, model definition, and training loop.
- The CIFAR-10 dataset consists of 10 classes, each containing 5,000 training images and 100 test images
    - total 50,000 train images, 1,000 test images.


## Used Language / Framworks / Library
- Python 3.9.18
- PyTorch
- CUDA 12.8
- Numpy
- OpenCV
- OS
- tqdm


## Details & Features
- Data Preprocessing: Normalization using mean and standard deviation
- Data Augmentation: Horizontal Flip, CutOut(Implemented but not used because resolution of data is too small)
- Models Implemented:
    - VGG-16
    - ResNet-18, ResNet-50
    - Preactivation ResNet
    - RoR-3 (Residual Network of Residaul Networks) - 110 layer Pre-Activation Version
- Training Utilities: Checkpoint save/load


## Performance
- CIFAR-10 Test Accuracy: 91% (100k iterations, minibatch  size: 128) (Using RoR-3 110 model)


## WIP
- DenseNet Implementation



### 한국어 설명

# 고급 딥러닝 프로그래밍 수업 프로젝트 - CIFAR10 Dataset 학습

## 상세설명
- 이 프로젝트의 목표는 CIFAR-10 이미지 분류에서 높은 정확도를 달성하는 것임.
- 데이터 전처리, 모델 정의, 학습 루프 등 전체 딥러닝 파이프라인을 구현함.
- CIFAR-10 데이터셋은 총 10개의 클래스(Class)로 구성되어 있으며, 각 클래스마다 5,000장의 학습 이미지와 100장의 테스트 이미지가 있음.
    - 전체데이터: 학습용 50,000장, 테스트용 1,000장


## 사용된 언어 / 프레임워크 / 라이브러리
- Python 3.9.18
- PyTorch
- CUDA 12.8
- Numpy
- OpenCV
- OS
- tqdm


## 세부 내용 및 특징
- 데이터 전처리: 평균 및 표준편차를 이용한 정규화
- 데이터 증강: 좌우 반전(FLip), CutOut(구현은 되어있지만 사용되지 않음. 32 * 32 이미지의 해상도가 너무 낮기때문)
- 구현한 모델:
    - VGG-16
    - ResNet-18, ResNet-50
    - PreActivation ResNet
    - RoR-3 (Residual Network of Residaul Networks) - 110 레이어 Pre-Activation 버전


## 성능
- CIFAR-10 테스트 정확도: 91% (65k iterations, 미니배치 크기: 128) (RoR-3 110 모델 사용)


## 진행중
- DenseNet 구현
