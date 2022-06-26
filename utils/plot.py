# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 12:16:39 2022

@author: User
"""
import matplotlib.pyplot as plt
import numpy as np
import os


def Plot_Val_Result_History(data_type,save_model,epochs,history_list,save_folder_dir,save_name):
    #==================================================================
    #==============plot recall at each epoch===========================
    
    if data_type == 0:
        data = "avg_precision"
    elif data_type==1:
        data = "avg_recall"
    elif data_type==2:
        data = "avg_accuracy"
    elif data_type==3:
        data = "avg_loss"
        
     
    print("plot ", data, " at each epoch :")
    print(epochs)
    print(history_list)
    num = 3
    plt.figure(figsize = (17,9))
    
    for a,b in zip(epochs, history_list):
        if data_type == 3:
            plt.text(a, b, str(b))
            if save_model==a:
                sm = b
        else:
            txt = float(int(b*10000))/float(100.0)
            if save_model==a:
                plt.text(a, b, str(txt)+'%,sm')
                sm = b
            if a%num == 0 or a==len(history_list):
                plt.text(a, b, str(txt)+'%')
            
    plt.plot(epochs,history_list)
    plt.xlabel('epochs')
    plt.ylabel(data)
    title_txt = str(data) + " at each epoch, sm: epoch =" + str(a) + " " +str(data) + "=" + str(sm)
    plt.title(title_txt)
    save_path = os.path.join(save_folder_dir,save_name)
    plt.savefig(save_path)
    plt.show()
    
    return sm
    #==================================================================