# ADL 5조.
# 개인적으로 분석한 정보.
# CIFAR-10 dataset은 32 * 32 사이즈의 컬러 이미지 (RGB 3채널)로 구성됨.
# test 이미지셋은 100장씩 존재함.
# 각각의 클래스 내에는 여러 종류의 사진이 존재함. 물론, 같은 종류의 클래스로 분류됨.
# train 이미지셋은 5000장씩 존재함.  airplane, automobile, ... truck 의 10개 클래스.
# 종합 5만장의 train image 존재.
# 불러올때 N, H, W, C 순서로 불러와지기 때문에 새롭게 정해야함.
# -> transpose를 사용해서 차원 변경.
# VGG, RESNET 모델을 구현함.

import numpy as np
import torch
import cv2
import os
from tqdm import tqdm
import time

# label = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#           'dog', 'frog', 'horse', 'ship', 'truck']

# image번호는 순서대로가 아님. 따라서 해당 클래스 내의 이미지면 모두 불러오게 만들기.
def load_image(path, num_img):
    imgs = np.zeros((num_img, 32, 32, 3)) # (N, H, W, C) 선언.
    cls = np.zeros(num_img)

    cls_names = sorted(os.listdir(path)) # 클래스 이름로드 ex) airplane, automobile, ... // sorted를 사용해서 알파벳 순서대로 정렬. -> 추가
    if path == './CIFAR10/test':
        print("\n------- Test Data Loading -------")
    else:
        print("\n------- Train Data Loading -------")
    img_count = 0

    for ic in tqdm(range(len(cls_names)), ncols=100):
        path_temp = path + '/' + cls_names[ic] + '/' # CIFAR10/train/airplane/ ... 형태
        # tqdm.write("Loading Class : %s" % cls_names[ic])
        img_names = os.listdir(path_temp) # 해당 클래스 내의 이미지 이름로드

        for im in range(len(img_names)):
            img = cv2.imread(path_temp + img_names[im]) # 이미지 불러오기. 
            # 색상 채널 변환 해줘야함. 
            # temp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV는 BGR로 불러오기 때문에 RGB로 변환.
            # 공지로 필요없어짐.

            # 임시로 temp의 shape 및 dtype 확인.
            # print(temp.shape, temp.dtype) # (32, 32, 3) uint8

            imgs[img_count, :, :, :] = img # (32, 32, 3) [H, W, C]

            cls[img_count] = ic
            img_count += 1
    print("---------------------------------\n")
    return imgs, cls

# 학습 테스트용 함수. 
def load_test_image(path, num_img):
    num = num_img / 10 # 클래스 개수로 나누기.
    imgs = np.zeros((num_img, 32, 32, 3)) # (N, H, W, C) 선언.
    cls = np.zeros(num_img)

    cls_names = sorted(os.listdir(path)) # 클래스 이름로드 ex) airplane, automobile, ... // sorted를 사용해서 알파벳 순서대로 정렬. -> 추가
    if path == './CIFAR10/test':
        print("\n------- Test Data Loading -------")
    else:
        print("\n------- Train Data Loading -------")
    img_count = 0

    for ic in range(len(cls_names)):
        path_temp = path + '/' + cls_names[ic] + '/' # CIFAR10/train/airplane/ ... 형태
        print("Loading Class : ", cls_names[ic])
        img_names = os.listdir(path_temp) # 해당 클래스 내의 이미지 이름로드

        for im in range(num):
            img = cv2.imread(path_temp + img_names[im]) # 이미지 불러오기.

            imgs[img_count, :, :, :] = img # (32, 32, 3) [H, W, C]

            cls[img_count] = ic
            img_count += 1
    print("---------------------------------\n")
    return imgs, cls




# train image가 여러개 존재할때, 이를 CutMix를 활용해서 랜덤하게 생성.
# def data_augmentation(image):
#     # CutMix

#     return image

# CutMix에서는 

# CutMix Github 참고함. trans pose는 Mini_batch_training 함수에서 진행중.
# lambda 는 논문에 나온 상수값.
# train_img의 경우에는 shape 가 [B, W, H, C] 기본설정.

# def rand_bbox(train_img, lam): 

#     W = train_img.shape[2]
#     return 0

# ------------- CUTOUT -----------------
# Cutout 기법을 활용해 데이터를 가공.
# Cutout Github 참고함.
# 핵심은 일정 부분을 0으로 만들어버리는 것. -> 데이터에 의도적인 손상을 주는 HOLE을 만듬.
def cut_out(image, holes, length):
    # image = [H, W, C]
    H = image.shape[0]
    W = image.shape[1]

    # 이미지를 가릴 mask 크기 설정.

    mask = np.ones((H, W, 1), np.float32) # (H, W) 크기의 1로 채워진 마스크 생성. 이미지와 같은 크기의 마스크임.

    for n in range(holes): # 구멍개수만큼 반복
        y = np.random.randint(H) # 0 ~ H-1
        x = np.random.randint(W) # 0 ~ W-1

        y1 = np.clip(y - length // 2, 0, H) # y좌표의 시작점
        y2 = np.clip(y + length // 2, 0, H) # y좌표의 끝점
        x1 = np.clip(x - length // 2, 0, W) # x좌표의 시작점
        x2 = np.clip(x + length // 2, 0, W) # x좌표의 끝점

        # 이걸 그림으로 표현하면 x와 y를 기준으로 length 만큼의 정사각형 범위를 0으로 만드는 것임.

        mask[y1: y2, x1: x2] = 0.0 # 해당 부분을 0으로 만듬.

    # 이제 계산을 해야하기 때문에, 이미지와 마스크를 같은 tensor로 만들어야함.
    output = image * mask

    return output # 0으로 된 부분은 이미지가 사라짐.
# ---------------------------------------


# 이미지 뒤집기(좌우반전 FTN)
def flip_image(image):
    return cv2.flip(image, 1) # -1, 0, 1


# 여기서 이미지 뒤집기와 CUTOUT 등 데이터 증강을 진행.
def Mini_batch_training(train_img , train_cls, batch_size):

    batch_img = np.zeros((batch_size, 32, 32, 3)) # (B, H, W, C)
    batch_cls = np.zeros(batch_size)

    rand_num = np.random.randint(0, train_img.shape[0], size = batch_size)

    for it in range(batch_size):
        temp = rand_num[it]
        img = train_img[temp]

        temp_prob = np.random.rand()
        # # Cutout 및 Flip 적용.
        # if temp_prob < 0.1:
        #     img = cut_out(img, holes=np.random.randint(1, 3), length=np.random.randint(2, 5)) # 8 * 8 크기의 구멍을 1개 뚫음.
        # elif 0.1 <= temp_prob < 0.5:
        #     # 반전 적용 상하, 좌우, 상하좌우 모두 반전.
        #     img = flip_image(img) 
        # else:
        #     # 그대로
        #     pass

        # # Without Cutout
        if temp_prob < 0.5:
            # 반전 적용 좌우 반전.
            img = flip_image(img)
        else:
            # 그대로
            pass

        batch_img[it, :, :, :] = img / 255.0 # image scaling 0 ~ 255 -> 0 ~ 1 or -1 ~ 1 normalization
        batch_cls[it] = train_cls[temp]
    
    batch_img = np.transpose(batch_img, (0, 3, 1, 2)) # (B, C, H, W) 형태로 바꿔주기
    
    return batch_img, batch_cls

# Neural Net work
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, hidden_size1) # 1st layer
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2) # 2nd layer
        self.fc3 = torch.nn.Linear(hidden_size2, output_size) # 3rd layer / output
        self.act_ReLU = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x) 
        x = self.act_ReLU(x)

        x = self.fc2(x)
        x = self.act_ReLU(x)

        x = self.fc3(x) # no activation function for output layer - will use softmax
        # but why? - we will use "SOFTMAX" cross entropy loss function

        return x
    
class CNN(torch.nn.Module):
    def __init__(self, output_size = 10):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3,padding=1) # RGB 3채널
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3,padding=1)

        self.fc1 = torch.nn.Linear(8 * 8 * 64, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 10)

        self.dropout = torch.nn.Dropout(p=0.5)

        self.MaxP = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.ReLU = torch.nn.ReLU()

    def forward(self, x):
        # x = [B, C, H, W] = 
        x = self.conv1(x) # 
        x = self.ReLU(x) # 
        x = self.MaxP(x) # 

        x = self.conv2(x) # 
        x = self.ReLU(x) # 
        x = self.MaxP(x) # [64, 64, 8, 8] -> [64, 8 * 8 * 64] // 64 * (8 * 8 * 64)

        x = torch.reshape(x, [-1, 8 * 8 * 64]) # [64, 8 * 8 * 64] batch, channel
        x = self.fc1(x)
        x = self.ReLU(x)

        x = self.fc2(x)
        x = self.ReLU(x)

        x = self.fc3(x)
        # no activation function for output layer - will use softmax

        return x
    
class CNN_v2(torch.nn.Module):
    def __init__(self, output_size = 10):
        super(CNN, self).__init__()

        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(32, 64, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(7 * 7 * 64, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256, 10)
        )


    def forward(self, x):
        x = self.covns(x)

        x = torch.reshape(x, [-1, 7 * 7 * 64]) # [64, 7 * 7 * 64] batch, channel

        x = self.mlp(x)
        # no activation function for output layer - will use softmax

        return x

# for CIFAR10
# 기본 예제를 CIFAR10에 맞게 수정.
class CNN_v3(torch.nn.Module):
    def __init__(self, output_size = 10):
        super(CNN_v3, self).__init__()

        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(32, 64, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(8 * 8 * 64, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(256, 10)
        )


    def forward(self, x):
        x = self.convs(x)

        x = torch.reshape(x, [-1, 8 * 8 * 64]) # Linear 형태로 변환.

        x = self.mlp(x)
        # no activation function for output layer - will use softmax

        return x
    
# Using VGG architecture
# VGG 풀 구조를 가져오긴 했지만, 학습이 제대로 되지 않는 모습을 확인.
class CNN_VGG(torch.nn.Module):
    def __init__(self, outputsize = 10):
        super(CNN_VGG, self).__init__()

        # self.input_layer = torch.nn.Sequential(
        #     torch.nn.Conv2d(3, 32, kernel_size=3,padding=1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(32, 32, kernel_size=3,padding=1),
        #     torch.nn.ReLU(),
        # )

        self.block1 = torch.nn.Sequential(
            # conv1-1
            torch.nn.Conv2d(3, 64, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            # conv1-2
            torch.nn.Conv2d(64, 64, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            # maxpool
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 32 -> 16
        )

        self.block2 = torch.nn.Sequential(
            # conv2-1
            torch.nn.Conv2d(64, 128, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            # conv2-2
            torch.nn.Conv2d(128, 128, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            # maxpool
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 16 -> 8
        )

        self.block3 = torch.nn.Sequential(
            # conv3-1
            torch.nn.Conv2d(128, 256, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            # conv3-2
            torch.nn.Conv2d(256, 256, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            # conv3-3
            torch.nn.Conv2d(256, 256, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            # maxpool
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 8 -> 4
        )

        self.block4 = torch.nn.Sequential(
            # conv4-1
            torch.nn.Conv2d(256, 512, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            # conv4-2
            torch.nn.Conv2d(512, 512, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            # conv4-3
            torch.nn.Conv2d(512, 512, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            # maxpool
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 4 -> 2
        )

        self.block5 = torch.nn.Sequential(
            # conv5-1
            torch.nn.Conv2d(512, 512, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            # conv5-2
            torch.nn.Conv2d(512, 512, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            # conv5-3
            torch.nn.Conv2d(512, 512, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            # maxpool
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 2 -> 1
        )

        self.full_connected_layer = torch.nn.Sequential(
            torch.nn.Linear(1 * 1 * 512, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(2048, outputsize)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = torch.reshape(x, [-1, 1 * 1 * 512]) # Linear 형태로 변환.

        x = self.full_connected_layer(x)

        return x

# VGG is too heavy for CIFAR10. 그래서 조금 가볍게 만듬.
# VGG 논문의 핵심은 depth가 증가할수록 성능이 좋아진다는 것.
# 하지만, 분석 결과, 32 * 32 사이즈에서는 너무 깊으면 오히려 제대로 학습이 안되는것을 확인.
# 따라서 block 개수를 줄임.
class CNN_VGG_small(torch.nn.Module):
    def __init__(self, outputsize = 10):
        super(CNN_VGG_small, self).__init__()

        self.block1 = torch.nn.Sequential(
            # conv1-1
            torch.nn.Conv2d(3, 64, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            # conv1-2
            torch.nn.Conv2d(64, 64, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            # maxpool
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 32 -> 16
        )

        self.block2 = torch.nn.Sequential(
            # conv2-1
            torch.nn.Conv2d(64, 128, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            # conv2-2
            torch.nn.Conv2d(128, 128, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            # maxpool
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 16 -> 8
        )

        self.block3 = torch.nn.Sequential(
            # conv3-1
            torch.nn.Conv2d(128, 256, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            # conv3-2
            torch.nn.Conv2d(256, 256, kernel_size=3,padding=1),
            torch.nn.ReLU(),
            # maxpool
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 8 -> 4
        )

        self.full_connected_layer = torch.nn.Sequential(
            torch.nn.Linear(4 * 4 * 256, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(2048, outputsize)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = torch.reshape(x, [-1, 4 * 4 * 256]) # Linear 형태로 변환.

        x = self.full_connected_layer(x)

        return x

# Using RESNET architecture
# VGG에서 우리가 알 수 있었듯, 깊이가 깊어질수록 더 좋은 성능이 나온다고 이야기함. 
# 하지만, 실험 결과 32 * 32 사이즈에서는 너무 깊어지면 오히려 성능이 떨어지는 것을 확인.
# 따라서, ResNet 구조를 활용해서 깊이를 늘려보는것도 좋을 것 같음.

## ResNet 18 구조를 활용. 
# RESNET 의 기본은 잔차연결. 수식적으로 표현을 하게된다면, F(x) = H(x) + x
# 기본 구조는 conv - relu - conv - add - relu
# 따라서 Conv - relu - conv 까지만 구조를 만들고 이후는 forward에서 잔차연결을 구현.
class CNN_RESNET(torch.nn.Module):
    def __init__(self, outputsize = 10):
        super(CNN_RESNET, self).__init__()

        self.ReLU = torch.nn.ReLU()

        self.input_layer = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
        )


        self.block1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )

        self.block1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )

        self.block2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.block2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.block3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )
        self.block3_2 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )

        self.block4_1 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )

        self.block4_2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )

        self.block5_1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )

        self.block5_2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )

        self.fullconnected_layer = torch.nn.Sequential(
            torch.nn.Linear(1 * 1 * 512, 1000),
            torch.nn.ReLU()
        )

        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=2, stride=2)
        self.conv5 = torch.nn.Conv2d(512, 512, kernel_size=2, stride=2)

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(1000, outputsize)
        )

    # RESNET 구조는 FORWARD에서 잔차연결이 들어가게 설계 할 예정임.
    def forward(self, x):
        # 입력레이어
        x = self.input_layer(x)

        # 잔차연결 시작. but stride 때문에 차원변경이 일어나기 떄문에, 해당 부분을 맞춰줘야함.
        residual = x
        x = self.block1_1(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block1_2(x)
        x += residual
        x = self.ReLU(x)

        residual = self.conv2(x)
        x = self.block2_1(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block2_2(x)
        x += residual
        x = self.ReLU(x)

        residual = self.conv3(x)
        x = self.block3_1(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block3_2(x)
        x += residual
        x = self.ReLU(x)

        residual = self.conv4(x)
        x = self.block4_1(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block4_2(x)
        x += residual
        x = self.ReLU(x)
        
        residual = self.conv5(x)
        x = self.block5_1(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block5_2(x)
        x += residual
        x = self.ReLU(x)

        x = torch.nn.AdaptiveAvgPool2d((1, 1))(x) # Global Average Pooling

        x = torch.reshape(x, [-1, 1 * 1 * 512]) # Linear 형태로 변환.

        x = self.fullconnected_layer(x)
        x = self.output_layer(x)

        return x
    
# ResNet 50 구조를 활용.
# ResNet 50의 핵심은 Bottleneck 구조.
# Bottleneck 구조는 conv1x1 - BN - ReLU - conv3x3 - BN - ReLU - conv1x1 - BN 구조로 이루어짐.
# BATCHNORM 또한 구현해야할 항목 중 하나.
class CNN_RESNET_50(torch.nn.Module):
    def __init__(self, outputsize = 10):
        super(CNN_RESNET_50, self).__init__()
        
        self.ReLU = torch.nn.ReLU()

        self.input_layer = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
        )

        # self.Maxpool33 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        # 해상도가 너무 작아져서 Maxpooling은 제거함.

### -------------------- conv2_x -> 3개
        self.block2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 256, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(256)
        )

        self.block2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 64, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 256, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(256)
        )

        self.blokc2_3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 64, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 256, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(256)
        )
### -------------------- end

### -------------------- conv3_x -> 4개
        self.block3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=1, stride=2, padding=0),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 512, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(512)
        )

        self.block3_2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 128, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 512, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(512)
        )

        self.block3_3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 128, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 512, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(512)
        )

        self.block3_4 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 128, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 512, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(512)
        )
### -------------------- end

### -------------------- conv4_x -> 6개
        self.block4_1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1, stride=2, padding=0),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 1024, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(1024)
        )

        self.block4_2 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 256, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 1024, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(1024)
        )
        
        self.block4_3 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 256, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 1024, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(1024)
        )

        self.block4_4 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 256, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 1024, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(1024)
        )

        self.block4_5 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 256, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 1024, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(1024)
        )

        self.block4_6 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 256, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 1024, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(1024)
        )
### -------------------- end

### -------------------- conv5_x -> 3개
        self.block5_1 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 512, kernel_size=1, stride=2, padding=0),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 2048, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(2048)
        )

        self.block5_2 = torch.nn.Sequential(
            torch.nn.Conv2d(2048, 512, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 2048, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(2048)
        )

        self.block5_3 = torch.nn.Sequential(
            torch.nn.Conv2d(2048, 512, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 2048, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(2048)
        )
### -------------------- end

        # projection conv
        self.proj2 = torch.nn.Conv2d(64, 256, kernel_size=1, stride=1)
        self.proj3 = torch.nn.Conv2d(256, 512, kernel_size=1, stride=2)
        self.proj4 = torch.nn.Conv2d(512, 1024, kernel_size=1, stride=2)
        self.proj5 = torch.nn.Conv2d(1024, 2048, kernel_size=1, stride=2)

        # Global Average Pooling
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # FC
        self.fullconnected_layer = torch.nn.Sequential(
            torch.nn.Linear(1 * 1 * 2048, 1000),
            torch.nn.ReLU()
            # torch.nn.Dropout(p=0.3),
            # torch.nn.Linear(2048, 1000),
            # torch.nn.ReLU() # 해당 내용은 VGG의 마지막 FC구조에서 차용함. 하지만, 학습능력이 떨어지는 모습을 보임.
        )

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(1000, outputsize)
        )

    def forward(self, x):
        x = self.input_layer(x)

        # x = self.Maxpool33(x)

### conv 2_x
        residual = self.proj2(x)
        x = self.block2_1(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block2_2(x)
        x += residual
        x = self.ReLU(x)
        
        residual = x
        x = self.blokc2_3(x)
        x += residual
        x = self.ReLU(x)
### end

### conv 3_x
        residual = self.proj3(x)
        x = self.block3_1(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block3_2(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block3_3(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block3_4(x)
        x += residual
        x = self.ReLU(x)
### end

### conv 4_x
        residual = self.proj4(x)
        x = self.block4_1(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block4_2(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block4_3(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block4_4(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block4_5(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block4_6(x)
        x += residual
        x = self.ReLU(x)
### end

### conv 5_x
        residual = self.proj5(x)
        x = self.block5_1(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block5_2(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block5_3(x)
        x += residual
        x = self.ReLU(x)
### end
        x = self.avgpool(x)

        x = torch.reshape(x, [-1, 1 * 1 * 2048]) # Linear 형태로 변환.

        x = self.fullconnected_layer(x)
        x = self.output_layer(x)

        return x
    
# 앞서 최초로 만들었던 RESNET 구조를 계승하는 모델. 50보다는 가벼운 18구조.
# 하지만, 이 구조에는 BatchNorm이 포함되어있음. 이전엔 없었음. 
class CNN_RESNET_18(torch.nn.Module):
    def __init__(self, outputsize = 10):
        super(CNN_RESNET_18, self).__init__()

        self.ReLU = torch.nn.ReLU()

        self.input_layer = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
        )


        self.block1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64)
        )

        self.block1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64)
        )

        self.block2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128)
        )

        self.block2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128)
        )

        self.block3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256)
        )
        self.block3_2 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256)
        )

        self.block4_1 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512)
        )

        self.block4_2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512)
        )

        self.block5_1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512)
        )

        self.block5_2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512)
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.fullconnected_layer = torch.nn.Sequential(
            torch.nn.Linear(1 * 1 * 512, 1000),
            torch.nn.ReLU()
        )


        # projection layer 
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=2, stride=2)
        self.conv5 = torch.nn.Conv2d(512, 512, kernel_size=2, stride=2)

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(1000, outputsize)
        )

    # RESNET 구조는 FORWARD에서 잔차연결이 들어가게 설계 할 예정임.
    def forward(self, x):
        # 입력레이어
        x = self.input_layer(x)

        # 잔차연결 시작. but stride 때문에 차원변경이 일어나기 떄문에, 해당 부분을 맞춰줘야함.
        residual = x
        x = self.block1_1(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block1_2(x)
        x += residual
        x = self.ReLU(x)

        residual = self.conv2(x) # 차원 변경을 위한 conv
        x = self.block2_1(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block2_2(x)
        x += residual
        x = self.ReLU(x)

        residual = self.conv3(x) # 차원 변경을 위한 conv
        x = self.block3_1(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block3_2(x)
        x += residual
        x = self.ReLU(x)

        residual = self.conv4(x) # 차원 변경을 위한 conv
        x = self.block4_1(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block4_2(x)
        x += residual
        x = self.ReLU(x)
        
        residual = self.conv5(x) # 차원 변경을 위한 conv
        x = self.block5_1(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block5_2(x)
        x += residual
        x = self.ReLU(x)

        x = self.avgpool(x) # Global Average Pooling

        x = torch.reshape(x, [-1, 1 * 1 * 512]) # Linear 형태로 변환.

        x = self.fullconnected_layer(x)
        x = self.output_layer(x)

        return x


# 이번 MEETING 에서 제안받은 다른 구조의 RESNET
# 구조는 대략 이런 방식임
# input x - BN - ReLU - conv - BN - ReLU - conv - add(residual) - output
# 블록은 RESNET-18을 계승.
class CNN_RESNET_NEW(torch.nn.Module):
    def __init__(self, outputsize = 10):
        super(CNN_RESNET_NEW, self).__init__()

        self.ReLU = torch.nn.ReLU()

        self.input_layer = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
        )

        self.block1_1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )

        self.block1_2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )

        self.block2_1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.block2_2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.block3_1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )

        self.block3_2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )

        self.block4_1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )

        self.block4_2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )

        self.block5_1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )

        self.block5_2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # Projection layer 
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=2, stride=2)
        self.conv5 = torch.nn.Conv2d(512, 512, kernel_size=2, stride=2)


        # Full Connected layer
        self.FC = torch.nn.Sequential(
            torch.nn.Linear(1 * 1 * 512, 1000),
            torch.nn.ReLU()
        )

        # Output layer
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(1000, outputsize)
        )


    def forward(self, x):
        x = self.input_layer(x)

        residual = x
        x = self.block1_1(x)
        x += residual

        residual = x
        x = self.block1_2(x)
        x += residual
        
        residual = self.conv2(x) # 차원 변경을 위한 conv
        x = self.block2_1(x)
        x += residual

        residual = x
        x = self.block2_2(x)
        x += residual

        residual = self.conv3(x) # 차원 변경을 위한 conv
        x = self.block3_1(x)
        x += residual

        residual = x
        x = self.block3_2(x)
        x += residual

        residual = self.conv4(x) # 차원 변경을 위한 conv
        x = self.block4_1(x)
        x += residual

        residual = x
        x = self.block4_2(x)
        x += residual

        residual = self.conv5(x) # 차원 변경을 위한 conv
        x = self.block5_1(x)
        x += residual

        residual = x
        x = self.block5_2(x)
        x += residual

        x = self.avgpool(x) # Global Average Pooling

        x = torch.reshape(x, [-1, 1 * 1 * 512]) # Linear 형태로 변환.

        x = self.FC(x)
        x = self.output_layer(x)
        
        return x
    
class CNN_RESNET_NEW_Light(torch.nn.Module):
    def __init__(self, outputsize = 10):
        super(CNN_RESNET_NEW_Light, self).__init__()

        self.ReLU = torch.nn.ReLU()

        self.input_layer = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
        )

        self.block1_1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )

        self.block1_2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )

        self.block2_1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.block2_2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.block3_1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )

        self.block3_2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )

        self.block4_1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )

        self.block4_2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )

        # self.block5_1 = torch.nn.Sequential(
        #     torch.nn.BatchNorm2d(512),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
        #     torch.nn.BatchNorm2d(512),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # )

        # self.block5_2 = torch.nn.Sequential(
        #     torch.nn.BatchNorm2d(512),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     torch.nn.BatchNorm2d(512),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # Projection layer 
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=2, stride=2)
        # self.conv5 = torch.nn.Conv2d(512, 512, kernel_size=2, stride=2)


        # Full Connected layer
        self.FC = torch.nn.Sequential(
            torch.nn.Linear(1 * 1 * 512, 1000),
            torch.nn.ReLU()
        )

        # Output layer
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(1000, outputsize)
        )


    def forward(self, x):
        x = self.input_layer(x)

        residual = x
        x = self.block1_1(x)
        x += residual

        residual = x
        x = self.block1_2(x)
        x += residual
        
        residual = self.conv2(x) # 차원 변경을 위한 conv
        x = self.block2_1(x)
        x += residual

        residual = x
        x = self.block2_2(x)
        x += residual

        residual = self.conv3(x) # 차원 변경을 위한 conv
        x = self.block3_1(x)
        x += residual

        residual = x
        x = self.block3_2(x)
        x += residual

        residual = self.conv4(x) # 차원 변경을 위한 conv
        x = self.block4_1(x)
        x += residual

        residual = x
        x = self.block4_2(x)
        x += residual

        # residual = self.conv5(x) # 차원 변경을 위한 conv
        # x = self.block5_1(x)
        # x += residual

        # residual = x
        # x = self.block5_2(x)
        # x += residual

        x = self.avgpool(x) # Global Average Pooling

        x = torch.reshape(x, [-1, 1 * 1 * 512]) # Linear 형태로 변환.

        x = self.FC(x)
        x = self.output_layer(x)
        
        return x


# 2중 RESNET 구조를 생각해보자.
# residual block 안에 residual block이 들어가는 형태.
# x - F(x) - F(x) + x - H(F(x) + x) - H(F(x) + x) + x

# x(input) - block1[F(x)] - residual term [F(x) + x] - block2[H(F(x)] + x) - residual term [H(F(x) + x) + x] - output

# 이런식으로 resnet 두개에 하나를 더 넣는 구조.
# 또한 dual resnet 구조에서 입력의 0.1~ 0.3 정도를 곱해서 더해주는 방식도 고려중
# -> 왜냐하면 너무 큰 값이 들어가게 된다면 오히려 성능이 떨어질 가능성을 생각. 잔상이 너무 강하게 남을수도 있으니까.

# 보류

class CNN_RESNET_DUAL_18(torch.nn.Module):
    def __init__(self, outputsize = 10):
        super(CNN_RESNET_DUAL_18, self).__init__()

        self.ReLU = torch.nn.ReLU()

        self.BN64 = torch.nn.BatchNorm2d(64)
        self.BN128 = torch.nn.BatchNorm2d(128)
        self.BN256 = torch.nn.BatchNorm2d(256)
        self.BN512 = torch.nn.BatchNorm2d(512)

        self.input_layer = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
        )


        self.block1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64)
        )

        self.block1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64)
        )

        self.block2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128)
        )

        self.block2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128)
        )

        self.block3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256)
        )
        self.block3_2 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256)
        )

        self.block4_1 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512)
        )

        self.block4_2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512)
        )

        self.block5_1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512)
        )

        self.block5_2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512)
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.fullconnected_layer = torch.nn.Sequential(
            torch.nn.Linear(1 * 1 * 512, 1000),
            torch.nn.ReLU()
        )


        # projection layer 
        self.conv2_over = torch.nn.Conv2d(64, 64, kernel_size=1, stride=1)

        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=2, stride=2)
        self.conv5 = torch.nn.Conv2d(512, 512, kernel_size=2, stride=2)

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(1000, outputsize)
        )

    # RESNET 구조는 FORWARD에서 잔차연결이 들어가게 설계 할 예정임.
    def forward(self, x):
        # 입력레이어
        x = self.input_layer(x)

        # 잔차연결 시작. but stride 때문에 차원변경이 일어나기 떄문에, 해당 부분을 맞춰줘야함.
        residual = x # 64채널
        residual_over = x
        x = self.block1_1(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block1_2(x) # 64채널
        x += residual
        x = self.ReLU(x)

        x += residual_over
        x = self.BN64(x)

        residual = self.conv2(x)
        x = self.block2_1(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block2_2(x)
        x += residual
        x = self.ReLU(x)

        residual = self.conv3(x)
        x = self.block3_1(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block3_2(x)
        x += residual
        x = self.ReLU(x)

        residual = self.conv4(x)
        x = self.block4_1(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block4_2(x)
        x += residual
        x = self.ReLU(x)
        
        residual = self.conv5(x)
        x = self.block5_1(x)
        x += residual
        x = self.ReLU(x)

        residual = x
        x = self.block5_2(x)
        x += residual
        x = self.ReLU(x)

        x = self.avgpool(x) # Global Average Pooling

        x = torch.reshape(x, [-1, 1 * 1 * 512]) # Linear 형태로 변환.

        x = self.fullconnected_layer(x)
        x = self.output_layer(x)

        return x
    

# Pyramidial Resnet구조에서 제안한 block
# 기본적으로 bottle neck 구조와 유사함.
