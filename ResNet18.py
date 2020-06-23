'''ResNet-18 Image classfication for cifar-10 with PyTorch
Author 'Sun-qian'.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import matplotlib.pyplot as plt
import numpy as np

savepath='features_pic'
if not os.path.exists(savepath):
    os.mkdir(savepath)

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),#padding=1可以保持特征图的大小不变
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),#inplace=True会改变输入的数据，
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut =nn.Sequential()#一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
        if stride != 1 or inchannel != outchannel:#虚线要加，实线不用加
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),#使用1*1大小的卷积核用于调整维度，见思维导图
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=4):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(#一个有序的容器，可以形成动态结构的神经网络
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),#参数含义：输入通道，输出通道，卷积核，步长，填充，输出是否加一个偏置
            nn.BatchNorm2d(64),#归一化，使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定，计算公式y=(x-mean(x))/(sqrt(Var(x))+eps),eps是一个很小的数，目的是为了计算的稳定性
            nn.ReLU(),#非线性化处理
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)#全连接运算

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))#执行block构造函数
            self.inchannel = channels
        return nn.Sequential(*layers)

    def draw_features(width, height, x, savename):
        tic = time.time()
        fig = plt.figure(figsize=(16, 16))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
        for i in range(width * height):
            plt.subplot(height, width, i + 1)
            plt.axis('off')
            # plt.tight_layout()
            img = x[0, i, :, :]
            pmin = np.min(img)
            pmax = np.max(img)
            img = (img - pmin) / (pmax - pmin + 0.000001)
            plt.imshow(img, cmap='gray')
            # print("{}/{}".format(i,width*height))
        fig.savefig(savename, dpi=100)
        fig.clf()
        plt.close()
        print("time:{}".format(time.time() - tic))

    def forward(self, x):#[1,3,32,32]
        out = self.conv1(x)#[1,64,32,32]
        out = self.layer1(out)#[1,64,32,32]#执行ResidualBlock的forward，会执行两次
        out = self.layer2(out)#[1,128,16,16]
        out = self.layer3(out)#[1,256,8,8]
        out = self.layer4(out)#[1,512,4,4]
        out = F.avg_pool2d(out, 4)#[1,512,1,1]
        out = out.view(out.size(0), -1)#[1,512]
        out = self.fc(out)#[1,4]

        return out#正值越大代表目标和卷积核越相关，负值意味着负相关。卷积核中的值当然可以为负。


def ResNet18():

    return ResNet(ResidualBlock)





