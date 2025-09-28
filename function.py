# 개인적으로 분석한 정보.
# 원본 CIFAR-10 DATASET 이 아닌 임의로 간추려진 DATASET임.
# CIFAR-10 dataset은 32 * 32 사이즈의 컬러 이미지 (RGB 3채널)로 구성됨.
# test 이미지셋은 100장씩 존재함.
# 각각의 클래스 내에는 여러 종류의 사진이 존재함. 물론, 같은 종류의 클래스로 분류됨.
# train 이미지셋은 5000장씩 존재함.  airplane, automobile, ... truck 의 10개 클래스.
# 종합 5만장의 train image 존재.
# 불러올때 N, H, W, C 순서로 불러와지기 때문에 새롭게 정해야함.
# transpose를 사용해서 차원 변경.

import numpy as np
import torch
import cv2
import os

from torchvision import v2

# label = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#           'dog', 'frog', 'horse', 'ship', 'truck']

# image번호는 순서대로가 아님. 따라서 해당 클래스 내의 이미지면 모두 불러오게 만들기.
def load_image(path, num_img):
    imgs = np.zeros((num_img, 32, 32, 3)) # (N, H, W, C) 선언.
    cls = np.zeros(num_img)

    cls_names = os.listdir(path) # 클래스 이름로드 ex) airplane, automobile, ...
    print("Class Names: ", cls_names)
    img_count = 0

    for ic in range(len(cls_names)):
        path_temp = path + '/' + cls_names[ic] + '/' # CIFAR10/train/airplane/ ... 형태
        print("Loading Class : ", cls_names[ic])
        img_names = os.listdir(path_temp) # 해당 클래스 내의 이미지 이름로드

        for im in range(len(img_names)):
            img = cv2.imread(path_temp + img_names[im]) # 이미지 불러오기. 
            # cv2.imshow('1', img)
            # cv2.waitKey(-1)
            # 색상 채널 변환 해줘야함. 
            imgs[img_count, :, :, :] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # (32, 32, 3) [H, W, C]
            
            cls[img_count] = ic
            img_count += 1

    return imgs, cls

# train image가 여러개 존재할때, 이를 CutMix를 활용해서 랜덤하게 데이터를 만들어냄. 
def data_augmentation(image):
    # CutMix, 

    return image

# class DataAugmentation(np.array): # N, H, W, C 순서로 input.
#     def __init__(self):
#         pass

#     def rotate(self, imgs):
#         return imgs

#     def CutMix(self, imgs):
#         return imgs



def Mini_batch_training(train_img , train_cls, batch_size):
    batch_img = np.zeros((batch_size, 32, 32, 3)) # (B, H, W, C)
    batch_cls = np.zeros(batch_size)

    rand_num = np.random.randint(0, train_img.shape[0], size = batch_size)

    for it in range(batch_size):
        temp = rand_num[it]
        batch_img[it, :, :, :] = train_img[temp, :, :, :] / 255.0 # image scaling 0 ~ 255 -> 0 ~ 1 or -1 ~ 1 normalization
        batch_cls[it] = train_cls[temp]
    
    batch_img = np.transpose(batch_img, (0, 3, 1, 2)) # (B, C, H, W) 형태로 바꿔주기
    
    return batch_img, batch_cls

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
    
# same as CNN class.
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
    
# Using RESNET architecture - now working
class CNN_v4(torch.nn.Module):
    def __init__(self, outputsize = 10):

