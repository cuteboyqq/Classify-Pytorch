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
from repVGG import *
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



def Get_Confusion_Matrix(y_true, y_pred,class_names):
    # 製作混淆矩陣
    cf_matrix = confusion_matrix(y_true, y_pred)                                # https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
    # 計算每個class的accuracy
    per_cls_acc = cf_matrix.diagonal()/cf_matrix.sum(axis=0)                    # https://stackoverflow.com/a/53824126/13369757
    #class_names = ['GreenLeft', 'GreenRight', 'GreenStraight','RedLeft','RedRight','YellowLeft','YellowRight','others']
    print(class_names)
    print(per_cls_acc)                                                          #顯示每個class的Accuracy
    #print("Plot confusion matrix")
    
    # 開始繪製混淆矩陣並存檔
    df_cm = pd.DataFrame(cf_matrix, class_names, class_names)    
    #ax1 = plt.subplot(1,2,1)
    plt.figure(figsize = (9,6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    if not os.path.exists("C:/TLR/confusion_matrix"):
        os.makedirs("C:/TLR/confusion_matrix")
    save_cm_path = "C:/TLR/confusion_matrix/"+ CM_FILENAME
    plt.savefig(save_cm_path)
    return cf_matrix

def Calculate_Precision_Recall_Accuracy(cf_matrix,epoch):
    TP = cf_matrix.diagonal()
    FP = cf_matrix.sum(0) - TP
    FN = cf_matrix.sum(1) - TP
    
    #avg_precision_list = []
    #avg_recall_list = []
        
    precision = TP / (TP + FP + 1e-12) 
    recall =   TP / (TP + FN + 1e-12)
    acc = [TP.sum()]*len(TP) / ([TP.sum()]*len(TP) + FP + FN)
    #avg_acc = per_cls_acc.mean()
    avg_precision = precision.mean()
    avg_recall = recall.mean()
    avg_acc = acc.mean()
    print("epoch = ",epoch)
    print("avg_precision =",avg_precision)
    print("avg_recall = ",avg_recall)
    print("avg_acc =",avg_acc)
    
def validate(test_loader, model, criterion, y_pred, y_true):
    model.eval()
    Total_loss = 0
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            #if args.gpu is not None:
            #images = images.cuda('cuda', non_blocking=True)
            if torch.cuda.is_available():
                images = images.cuda('cuda', non_blocking=True)
                target = target.cuda('cuda', non_blocking=True)
            else:
                images = images
                target = target

            #output = model(images)
            output = F.softmax(model(images)).data
            _, preds = torch.max(output, 1)                            # preds是預測結果
            loss = criterion(output, target)
            Total_loss = Total_loss + loss
            #print("loss =", loss)
            y_pred.extend(preds.view(-1).detach().cpu().numpy())       # 將preds預測結果detach出來，並轉成numpy格式       
            y_true.extend(target.view(-1).detach().cpu().numpy())      # target是ground-truth的label
            
    return y_pred, y_true, Total_loss


def Do_Validation_At_Each_Epoch(SAVE_MODEL_PATH,
                                VAL_DATA_DIR,
                                IMAGE_SIZE,
                                ENABLE_VALIDATION,
                                y_pred,
                                y_true,
                                epoch):
    '''
    =======================================================
    Start do validation 
    =======================================================
    '''
    modelPath = SAVE_MODEL_PATH
    if os.path.exists(modelPath):
        MODEL_EXIST = True
    else:
        MODEL_EXIST = False
    if ENABLE_VALIDATION and MODEL_EXIST:
        
        #IMAGE_SIZE = 32
        #modelPath = r"C:/TLR/model/TLR_ResNet18-2022-04-14-Size32-4-8-16-32.pt"
        size = (IMAGE_SIZE,IMAGE_SIZE)
        img_test_data = torchvision.datasets.ImageFolder(VAL_DATA_DIR,
                                                    transform=transforms.Compose([
                                                        transforms.Resize(size),
                                                        #transforms.RandomHorizontalFlip(),
                                                        #transforms.Scale(64),
                                                        transforms.CenterCrop(size),
                                                        transforms.ToTensor()
                                                        ])
                                                    )
        
        print(len(img_test_data))
        '''
        ============================================================================
        Do validation
        function : torch.utils.data.DataLoader
        load the images to tensor
        ============================================================================
        '''
        BATCH_SIZE_VAL = BATCH_SIZE
        test_loader = torch.utils.data.DataLoader(img_test_data, batch_size=BATCH_SIZE_VAL,shuffle=False,drop_last=False)
        print(len(test_loader))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')
        model = torch.load(modelPath).to(device)
        criterion = nn.CrossEntropyLoss()
        
        y_pred, y_true, val_loss = validate(test_loader, model, criterion, y_pred,y_true)
        '''
        =============================================================================
        validation : confusion matrix
        =============================================================================
        '''
        cf_matrix = Get_Confusion_Matrix(y_true, y_pred,class_names)
        
        '''
        =============================================================================
        validation : precision, recall, acc
        =============================================================================
        '''
        Calculate_Precision_Recall_Accuracy(cf_matrix,epoch)

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
                                       epoch);
        
        
        
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
    net = RepVGG(num_blocks=[2, 2, 2, 2], num_classes=8,
                  width_multiplier=[0.25, 0.25, 0.25, 0.25], override_groups_map=None, deploy=False)
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
                 