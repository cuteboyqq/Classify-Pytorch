# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 01:50:36 2022

@author: User
"""
import torch.nn as nn
import torch.nn.functional as F
 
#定义残差块ResBlock
class ResBlock(nn.Module):
     def __init__(self, inchannel, outchannel, stride=1):
         super(ResBlock, self).__init__()
         #这里定义了残差块内连续的2个卷积层
         self.left = nn.Sequential(
             nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
             nn.BatchNorm2d(outchannel),
             nn.ReLU(inplace=True),
             nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
             nn.BatchNorm2d(outchannel)
         )
         self.shortcut = nn.Sequential()
         if stride != 1 or inchannel != outchannel:
             #shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
             self.shortcut = nn.Sequential(
                 nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                 nn.BatchNorm2d(outchannel)
             )
             
     def forward(self, x):
         out = self.left(x)
         #将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
         out = out + self.shortcut(x)
         out = F.relu(out)
         
         return out
 
class ResNet(nn.Module):
    def __init__(self, ResBlock,c1,c2,c3,c4,num_classes=8):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, c1, 2, stride=1)#16
        self.layer2 = self.make_layer(ResBlock, c2, 2, stride=2)#32
        self.layer3 = self.make_layer(ResBlock, c3, 2, stride=2)#64        
        self.layer4 = self.make_layer(ResBlock, c4, 2, stride=2)#128        
        self.fc = nn.Linear(c4, num_classes)#512 for 64*64,128 for 32*32
    #这个函数主要是用来，重复同一个残差块    
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        #在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        #print(out.shape)
        out = self.layer1(out)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = self.layer4(out)
        #print(out.shape)
        out = F.avg_pool2d(out, 4)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc(out)
        #print(out.shape)
        return out