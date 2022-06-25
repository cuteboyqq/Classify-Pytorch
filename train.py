# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 23:46:07 2022

@author: User
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import torch
import torchvision
from torchvision import transforms, utils
#import tkinter
#from tkinter import *
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import torch.nn as nn
from network.repVGG import *
from network.resnet import *
from utils.val import *
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import os
import random
import shutil
import time
import json
import warnings
import torch.nn.functional as F



def train(net,
          nums_epoch,
          IMAGE_SIZE,
          BATCH_SIZE,
          SAVE_MODEL_PATH,
          TRAIN_DATA_DIR,
          VAL_DATA_DIR,
          ENABLE_VALIDATION,
          class_names,
          CM_FILENAME):
    
    
    size = (IMAGE_SIZE,IMAGE_SIZE)
    img_data = torchvision.datasets.ImageFolder(TRAIN_DATA_DIR,
                                                transform=transforms.Compose([
                                                    transforms.Resize(size),
                                                    #transforms.RandomHorizontalFlip(),
                                                    #transforms.Scale(64),
                                                    transforms.CenterCrop(size),
                                                 
                                                    transforms.ToTensor()
                                                    ])
                                                )
    
    data_loader = torch.utils.data.DataLoader(img_data, batch_size=BATCH_SIZE,shuffle=True,drop_last=False)
    print(len(data_loader))
    
    '''============================================================================================================================================'''  
    classes = ('GreenLeft', 'GreenRight', 'GreenStraight','RedLeft','RedRight','YellowLeft','YellowRight','others')
    #classes = ('Green', 'GreenLeft', 'GreenRight', 'GreenStraight', 'Off', 'others', 'Red', 'RedLeft', 'RedRight', 'Yellow', 'YellowLeft','YellowRight')
    # 显示一张图片
    def imshow(img):
        img = img / 2 + 0.5   # 逆归一化
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    
    # 任意地拿到一些图片
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    
    # 显示图片
    imshow(torchvision.utils.make_grid(images))
    # 显示类标
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    def show_batch(imgs):
        grid = utils.make_grid(imgs,nrow=5)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.title('Batch from dataloader')
    
    for i, (batch_x, batch_y) in enumerate(data_loader):
        if(i<6):
            print(i, batch_x.size(), batch_y.size())
    
            show_batch(batch_x)
            plt.axis('off')
            plt.show()
            
    for i, data in enumerate(data_loader):
      img,label=data
      print(i," : ",label)
      
    '''
    =======================================================
    Parameter settings : criterion, optimizer, device
    =======================================================
    '''
    import torch.optim as optim
    '''loss function'''
    criterion = nn.CrossEntropyLoss()
    ''' optimizer method '''
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _lowest_loss = 1000.0
    
    '''
    =======================================================
    Start Train model at each epoch 
    =======================================================
    '''
    for epoch in range(nums_epoch):
        total_loss = 0.0
        tot_loss = 0.0 
        _loss = 0.0
        train_preds = []
        train_trues = []
        y_pred = []   #保存預測label
        y_true = []   #保存實際label
        
        for i, (inputs, labels) in enumerate(data_loader, 0):
            '''get batch images and corresponding labels'''
            inputs, labels = inputs.to(device), labels.to(device)
            '''initial optimizer to zeros'''
            optimizer.zero_grad()
            ''' put batch images to convolution neural network '''
            outputs = net(inputs)
            """calculate loss by loss function"""
            loss = criterion(outputs, labels)
            '''after calculate loss, do back propogation'''
            loss.backward()
            '''optimize weight and bais'''
            optimizer.step()
              
            _loss += loss.item()
            tot_loss += loss.data
            total_loss += loss.item()
            train_outputs = outputs.argmax(dim=1)
            
            train_preds.extend(train_outputs.detach().cpu().numpy())
            train_trues.extend(labels.detach().cpu().numpy())
            '''
            =======================================================
            
            After some epochs , Save the model which loss is lowest 
            (Noted, not save model currently, just to show loss info.)
            
            =======================================================
            '''
            if i % 6 == 0 and i > 0:  # 每3步打印一次损失值
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, _loss / 1))
                #if epoch > 0:
                '''    
                if _loss < _lowest_loss:
                    _lowest_loss = _loss
                    print('Start save model !')
                    torch.save(net, PATH)
                    print('save model complete with loss : %.3f' %(_loss))
                '''
                _loss = 0.0
        '''
        ==========================================================
        Save model if loss is the smallest at each epoch
        ==========================================================
        '''
        if tot_loss < _lowest_loss:
            save_model = epoch+1
            _lowest_loss = tot_loss
            print('Start save model !')
            torch.save(net, SAVE_MODEL_PATH)
            print('save model complete with loss : %.3f' %(tot_loss))
        #epochs.extend([epoch+1])
        
        '''
        =======================================================
        Start do validation at each epoch
        =======================================================
        '''
        Do_Validation_At_Each_Epoch(SAVE_MODEL_PATH,
                                       VAL_DATA_DIR,
                                       IMAGE_SIZE,
                                       ENABLE_VALIDATION,
                                       y_pred,
                                       y_true,
                                       epoch,CM_FILENAME,BATCH_SIZE,class_names);
        
        
        
if __name__=="__main__":
    
    TRAIN_DATA_DIR = r"C:\TLR\datasets\roi"
    VAL_DATA_DIR = r"C:\TLR\datasets\roi-test"
    IMAGE_SIZE = 32
    BATCH_SIZE = 300
    SAVE_MODEL_PATH = r"C:\TLR\model\repVGG_32.pt"
    nums_epoch = 50
    ENABLE_VALIDATION = True
    CM_FILENAME = "repVGG_32_8cls_CM.png"
    class_names = ['GreenLeft', 'GreenRight', 'GreenStraight','RedLeft','RedRight','YellowLeft','YellowRight','others']
    #net = RepVGG(num_blocks=[2, 2, 2, 2], num_classes=8,
    #              width_multiplier=[0.25, 0.25, 0.25, 0.25], override_groups_map=None, deploy=False)
    net = ResNet(ResBlock,8,16,32,64)
    if torch.cuda.is_available():
        net.cuda() 
        
    train(net,
          nums_epoch,
          IMAGE_SIZE,
          BATCH_SIZE,
          SAVE_MODEL_PATH,
          TRAIN_DATA_DIR,
          VAL_DATA_DIR,
          ENABLE_VALIDATION,
          class_names,
          CM_FILENAME)
                 