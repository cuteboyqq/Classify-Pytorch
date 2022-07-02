# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 02:04:30 2022

@author: User
"""

import torch
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import torch.nn as nn
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import time
import torch.nn.functional as F
import math
from  utils.plot import *
rep_dir = 'C:/RepVGG'
val_precision_folder_dir = rep_dir + '/plot_graph/precision/val'
val_recall_folder_dir = rep_dir + '/plot_graph/recall/val'
train_loss_dir = rep_dir + '/plot_graph/loss/train'
val_loss_dir = rep_dir + '/plot_graph/loss/val'
val_acc_dir = rep_dir + '/plot_graph/acc/val'
model_dir = rep_dir + '/model'
confusion_matrix_dir = rep_dir + '/confusion_matrix/'
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(val_precision_folder_dir):
    os.makedirs(val_precision_folder_dir)
    
if not os.path.exists(val_recall_folder_dir):
    os.makedirs(val_recall_folder_dir)
    
if not os.path.exists(train_loss_dir):
    os.makedirs(train_loss_dir)
    
if not os.path.exists(val_loss_dir):
    os.makedirs(val_loss_dir)

if not os.path.exists(val_acc_dir):
    os.makedirs(val_acc_dir)

def Get_Confusion_Matrix(y_true, y_pred,class_names,CM_FILENAME):
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
    if not os.path.exists(confusion_matrix_dir):
        os.makedirs(confusion_matrix_dir)
    save_cm_path = confusion_matrix_dir + CM_FILENAME
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
    return avg_precision, avg_recall, avg_acc
    
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
                                epoch,
                                CM_FILENAME,
                                BATCH_SIZE,
                                class_names,
                                avg_precision_list,
                                avg_recall_list,
                                val_loss_list,
                                avg_acc_list,
                                epochs,
                                save_model,
                                c1,c2,c3,c4,
                                date):
    
    '''
    ======================================================
    validation : define plot file name  at each train epoch
    ======================================================
    '''
    ch = str(c1)+'-'+ str(c2) +'-'+ str(c3) +'-'+ str(c4) #set~~~~~~~~~
    PLOT_PRECISION_NAME = "avg_precision" + date + 'Size' + str(IMAGE_SIZE) +'-' +ch + ".png"
    PLOT_RECALL_NAME = 'avg_recall' + date + 'Size' + str(IMAGE_SIZE) +'-' +ch + ".png"
    #PLOT_TRAIN_LOSS_NAME = ' _Loss_Epoch' + date + 'Size' + str(IMAGE_SIZE) +'-' + ch + ".png"
    PLOT_VAL_LOSS_NAE = 'Val_Loss_Epoch' + date + 'Size' + str(IMAGE_SIZE) + '-'+ch + ".png"
    PLOT_ACC_NAME = 'avg_acc' + date + 'Size' + str(IMAGE_SIZE) +'-' + ch + ".png"
    
    
    '''
    =======================================================
    validation : Start do validation 
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
        cf_matrix = Get_Confusion_Matrix(y_true, y_pred,class_names,CM_FILENAME)
        
        '''
        =============================================================================
        validation : precision, recall, acc
        =============================================================================
        '''
        avg_precision, avg_recall, avg_acc = Calculate_Precision_Recall_Accuracy(cf_matrix,epoch)
        
        '''add avg_pre, avg_recall, val_loss to list'''
        avg_precision_list.extend([avg_precision])
        avg_recall_list.extend([avg_recall])
        val_loss_list.extend(val_loss.view(-1).detach().cpu().numpy())
        avg_acc_list.extend([avg_acc])
        '''
        ==========================================================================
        validation : plot precision, recall, acc, loss at each epoch 
        0:precision, 1:recall, 2:acc, 3:val loss
        ==========================================================================
        '''
        '''plot precision/epoch'''
        sm_pre = Plot_Val_Result_History(0,save_model,epochs,avg_precision_list,val_precision_folder_dir,PLOT_PRECISION_NAME)
        '''plot recall/epoch'''
        sm_recall = Plot_Val_Result_History(1,save_model,epochs,avg_recall_list,val_recall_folder_dir,PLOT_RECALL_NAME)
        '''plot val loss/epoch'''
        sm_ValLoss = Plot_Val_Result_History(3,save_model,epochs,val_loss_list,val_loss_dir,PLOT_VAL_LOSS_NAE)
        '''plot val acc/epoch'''
        sm_acc = Plot_Val_Result_History(2,save_model,epochs,avg_acc_list,val_acc_dir,PLOT_ACC_NAME)
        #===================================================================
        #print("TP: {}, FP: {}, FN: {}".format(TP,FP,FN))
        #print("precision = TP/(TP + FP) :")
        #print("{}".format(precision))
        #print("recall = TP/(TP + FN) :")
        #print("{}".format(recall))
        print("avg prcision = {}".format(avg_precision))
        print("avg recall = {}".format(avg_recall))
        
        
        return avg_precision_list, avg_recall_list, val_loss_list, avg_acc_list, sm_pre, sm_recall, sm_ValLoss, sm_acc