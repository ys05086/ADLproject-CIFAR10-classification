# New function file
# file was too big.
# changes: to build Network easily, added ResNet block(Pre-activaation type)
#          made RoR-3 network (110)
#          added data preprocessing ftn ( for data normalization ) : subtract mean, divide std
#          made cutout and flip work in one ftn
#          changed data preprocessing to / 255.0 
#          made if clauses short ex) num_img // 10 if for_test else num_img
#          added data_preprocessing_ ftn 
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm

# Ftn for Image Loading // Added Test Data Setting 
def load_image(path, num_img, for_test = False, mean = None, std = None):
    # this is more minimal: 
    doncare = 10 if for_test else 1
    imgs = np.zeros((num_img // doncare, 32, 32, 3)) # shape : [N, H, W, C]
    cls = np.zeros(num_img // doncare)

    cls_names = sorted(os.listdir(path)) # class name loading

    print("\nTest Data Loading..." if path[-5:] == 'test/' else "\nTrain Data Loading...")

    img_count = 0

    for ic in tqdm(range(len(cls_names)), ncols = 100, desc = 'Loading Progress'): # class loop
        path_temp = path + '/' + cls_names[ic] + '/'
        img_names = os.listdir(path_temp) # file name loading(in each class folder)

        for im in range(len(img_names) // doncare): # if test mode enabled, load only 1/10 of images
            img = cv2.imread(path_temp + img_names[im]) # load image

            imgs[img_count, :, :, :] = img # [32, 32, 3] [H, W, C]
            cls[img_count] = ic # class index assignment
            img_count += 1
    # imgs = data_preprocessing(imgs) # data preprocessing
    imgs, mean, std = data_preprocessing_(imgs, mean, std)
    print("Image Loaded.")
    return imgs, cls, mean, std

# Ftn for Data Preprocessing
# Subtract mean, diveide std
def data_preprocessing(images): # [N, H, W, C] - numpy array
    # mean = np.mean(images, axis=(0, 1, 2)) # mean for each channel
    # std = np.std(images, axis=(0, 1, 2)) # std for each channel

    # images = (images - mean) / std # data preprocessing which we call normalization
    images = images / 255.0
    return images

def data_preprocessing_(images, mean = None, std = None):
    if mean is None or std is None:
        mean = np.mean(images, axis=(0, 1, 2)) # mean for each channel
        std = np.std(images, axis=(0, 1, 2)) # std for each channel

    images = (images - mean) / std # data preprocessing
    return images, mean, std

# Ftn for Image Augementation
def img_augmentation(image, cutout = False, flip = True, flip_prob = 0.5):
    if cutout:
        H = image.shape[0] # shape of image : [H, W, C]
        W = image.shape[1]
        holes = np.random.randint(1, 3) # number of holes
        length = np.random.randint(5, 10) # length of holes

        mask = np.ones((H, W, 1), np.float32)
        for n in range(holes):
            y = np.random.randint(H) # y center of hole
            x = np.random.randint(W) # x center of hole

            # calculating the top, bottom, left, right of the square
            y1 = np.clip(y - length // 2, 0, H) # start point of y axis
            y2 = np.clip(y + length // 2, 0, H) # end point of y axis
            x1 = np.clip(x - length // 2, 0, W) # start point of x axis
            x2 = np.clip(x + length // 2, 0, W) # end point of x axis

            mask[y1: y2, x1: x2] = 0.0

        image = image * mask # applying cutout on the image
    if flip:
        prob = np.random.rand() # random number between 0 and 1
        image = np.flip(image, 1) if prob > flip_prob else image
    return image

# Ftn for Mini-batch Training
def mini_batch_training(train_img, train_cls, batch_size, cutout = False, flip = True):
    batch_img = np.zeros((batch_size, 32, 32, 3)) # shape : [B, H, W, C]
    batch_cls = np.zeros(batch_size)

    rand_num = np.random.randint(0, train_img.shape[0], size = batch_size) # random number for mini-batch

    for it in range(batch_size):
        temp = rand_num[it]
        img = train_img[temp]

        batch_img[it, :, :, :] = img_augmentation(img, cutout=cutout, flip=flip) # image augmentation
        batch_cls[it] = train_cls[temp]

    batch_img = np.transpose(batch_img, (0, 3, 1, 2)) # (B,H,W,C) -> (B,C,H,W)
    return batch_img, batch_cls

# Network
# ResNet
# Wip but i think will not use - RoR is better
class ResNet(torch.nn.Module):
    def __init__(self, outputsize = 10):
        super(ResNet, self).__init__()
        self.block1_1 = Block(3, 16, stride_true=True)

# RoR-3 (Actually preactivation ror) -110
class RoRNet(torch.nn.Module):
    def __init__(self, outputsize = 10):
        super(RoRNet, self).__init__()
        self.in_channel = 16

        self.ReLU = torch.nn.ReLU()

        self.input_conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

        self.stage1 = RoRBlock(16, 16, stride=1, stride_true=False) # 16
        self.stage2 = RoRBlock(16, 32, stride=2, stride_true=True) # 32
        self.stage3 = RoRBlock(32, 64, stride=2, stride_true=True) # 64

        self.bn = torch.nn.BatchNorm2d(64)
        self.avgpool = torch.nn.AvgPool2d(8)

        # level 1 projection layer
        self.projection = torch.nn.Conv2d(16, 64, kernel_size=1, stride=4, bias=False)
        self.fc = torch.nn.Linear(64, outputsize)

    def forward(self, x):
        x = self.input_conv(x)
        residual = x
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x + self.projection(residual)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.avgpool(x)
        x = torch.reshape(x, [-1, 64]) # to linear
        x = self.fc(x)
        return x

# RoR-3 -146

# Residual Block (Pre Activation)
class Block(torch.nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1, stride_true = False):
        super(Block, self).__init__()

        self.ReLU = torch.nn.ReLU()
        self.stride_true = stride_true
        self.stride = stride
        self.conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = self.stride, padding = 1, bias=False)
        self.conv2 = torch.nn.Conv2d(out_channel, out_channel, kernel_size = 3, stride = 1, padding = 1, bias=False)

        self.bn1 = torch.nn.BatchNorm2d(in_channel)
        self.bn2 = torch.nn.BatchNorm2d(out_channel)

        # layer for projection
        self.projection = torch.nn.Conv2d(in_channel, out_channel, kernel_size = 1, stride = 2, bias=False)

    def forward(self, x):
        residual = self.projection(x) if self.stride_true else x

        x = self.bn1(x)
        x = self.ReLU(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.ReLU(x)
        x = self.conv2(x)

        x = x + residual
        return x

# RoR Block
class RoRBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1, stride_true = False):
        super(RoRBlock, self).__init__()

        self.stride_true = stride_true
        self.stride = stride

        self.block1 = Block(in_channel, out_channel, stride, stride_true) # first block has alwasy stride 2 and true
        self.block2 = Block(out_channel, out_channel, 1, False)
        self.block3 = Block(out_channel, out_channel, 1, False)
        self.block4 = Block(out_channel, out_channel, 1, False)
        self.block5 = Block(out_channel, out_channel, 1, False)
        self.block6 = Block(out_channel, out_channel, 1, False)
        self.block7 = Block(out_channel, out_channel, 1, False)
        self.block8 = Block(out_channel, out_channel, 1, False)
        self.block9 = Block(out_channel, out_channel, 1, False)
        self.block10 = Block(out_channel, out_channel, 1, False)
        self.block11 = Block(out_channel, out_channel, 1, False)
        self.block12 = Block(out_channel, out_channel, 1, False)
        self.block13 = Block(out_channel, out_channel, 1, False)
        self.block14 = Block(out_channel, out_channel, 1, False)
        self.block15 = Block(out_channel, out_channel, 1, False)
        self.block16 = Block(out_channel, out_channel, 1, False)
        self.block17 = Block(out_channel, out_channel, 1, False)
        self.block18 = Block(out_channel, out_channel, 1, False)

        # level 2 projcetion layer
        self.projection = torch.nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.projection_stride = torch.nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=False)

    def forward(self, x):
        residual = x

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)

        x = x + self.projection_stride(residual) if self.stride_true else x + self.projection(residual)

        return x
    
# BottleNeck Block
class BottleNeck(torch.nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1, stride_true = False):
        super(BottleNeck, self).__init__()

        self.ReLU = torch.nn.ReLU()
        self.stride_true = stride_true
        self.stride = stride
        self.neck1 = torch.nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=1, bias=False)
        self.conv = torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.neck2 = torch.nn.Conv2d(out_channel, out_channel, kernel_size=1, padding=1, bias=False)

        self.bn1 = torch.nn.BatchNorm2d(in_channel)
        self.bn2 = torch.nn.BatchNorm2d(out_channel)

        # layer for projection
        self.projection = torch.nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=False)

    def forward(self, x):
        residual = self.projection(x) if self.stride_true else x

        

        return x
