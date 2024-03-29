# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 23:54:12 2022

@author: User
"""
import torch
import torchvision
from torchvision import transforms, utils
import numpy as np
import os
import torch.nn.functional as F
from PIL import Image
import cv2
import glob


def main(repo_dir):
    import yaml
    inference_cfg_yaml_path = repo_dir + '/yaml/inference_cfg.yaml'
    with open(inference_cfg_yaml_path,'r') as f:
        data = yaml.safe_load(f)
        print(yaml.dump(data))
        
    #GET_INFER_ACCURACY = data['GET_INFER_ACCURACY']
    #IMAGE_SIZE = data['IMAGE_SIZE']
    #imagedir = data['imagedir']
    #modelPath = data['modelPath']
    c1,c2,c3,c4=data['channel'][0],data['channel'][1],data['channel'][2],data['channel'][3]
    ch = str(c1)+'-'+ str(c2) +'-'+ str(c3) +'-'+ str(c4) #set~~~~~~~~~
    #date = data['date']
    pred_dir = repo_dir + "/inference/TLR_" + data['net'] + data['date'] + 'Size' + str(data['IMAGE_SIZE']) + '-' + ch + '_result'
   
    Inference(data['imagedir'], data['modelPath'],pred_dir,data['IMAGE_SIZE']) 
   
    PREDICT_RESULT_IMG = pred_dir
    #class_names = data['class_names']
    
    if data['GET_INFER_ACCURACY']:  
        imagedir = PREDICT_RESULT_IMG
        #class_list = ["GreenLeft", "GreenRight", "GreenStraight","RedLeft",
        #              "RedRight","YellowLeft","YellowRight","others"]
        #class_list = class_names
        Calculate_Inference_Accuracy(imagedir,data['class_names'])
   
    
    COLLECT_FP_IMAGES = True
    import glob
    import os
    import shutil
    import tqdm
    if COLLECT_FP_IMAGES:
        
        def Analysis_img_path(img_path):
            img = os.path.basename(img_path)
            GT = img.split("_")[0]
            predict = img.split("_")[1]
            return GT, predict
        
        save_dir = data['save_fp_img_dir']
        if not os.path.exists(save_dir):os.makedirs(save_dir)
        search_dir = PREDICT_RESULT_IMG
        image_path_list = glob.iglob(os.path.join(search_dir,'**','*.jpg'))
        
        pbar = tqdm.tqdm(image_path_list)
        PREFIX = colorstr("Search FP images")
        pbar.desc = f'{PREFIX}'
        for img_path in pbar:
            #print(img_path)
            GT, predict = Analysis_img_path(img_path)
            #print(GT," ",predict)
            if not predict == GT:
                save_label_dir = os.path.join(save_dir,GT)
                if not os.path.exists(save_label_dir):os.makedirs(save_label_dir)
                shutil.copy(img_path,save_label_dir) 

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']
'''
=======================================================================================================================================
FUNCTION : Inference

infernece images
INPUT:
    imagedir:
        the image folder dir
    modelPath :
        the model path
=========================================================================================================================================
'''
def Inference(imagedir, modelPath,pred_dir,IMAGE_SIZE):
    rep_dir = os.path.dirname(os.path.realpath(__file__))
    classes_txt_path = rep_dir + '/classes.txt'
    with open(classes_txt_path, "r") as f:
        classes = f.read().split("\n")
   
        
    inputSize = (IMAGE_SIZE,IMAGE_SIZE)

    data_transforms_test = transforms.Compose([
                        #transforms.ToPILImage(),
                        transforms.Resize(inputSize),
                        transforms.CenterCrop(inputSize),
                        transforms.ToTensor()
                        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = torch.load(modelPath).to(device)
    model.eval()
    #image = Image.open(urlopen(imagePath)).convert('RGB')
    #image = Image.open(imagePath).convert('RGB')
    #image = cv2.imread(imagePath)
    search_img_dir = imagedir + "/**/*.jpg"
    img_list = glob.iglob(search_img_dir)
    img_count = 0
    y_pred = []   #保存預測label
    y_true = []   #保存實際label
    for img_path in img_list:
        GT = os.path.basename(os.path.dirname(img_path))
        img = cv2.imread(img_path) # opencv開啟的是BRG
        #cv2.imshow("OpenCV",img)
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        #image.show()
        #cv2.waitKey()
    
    
        image_tensor = data_transforms_test(image).float()
        image_tensor = image_tensor.unsqueeze_(0).to(device)
        output = F.softmax(model(image_tensor)).data.cpu().numpy()[0]
        prediction = classes[np.argmax(output)]
        pre_label = np.argmax(output)
        score = max(output)
        print(score, prediction," ",GT)
        result_img_name = prediction + "_" + str(score)
        
        '''
        =================================================================================
        '''
        #y_pred.extend(str(prediction))
        #y_true.extend(str(GT))
        #print(len(y_pred))
        #print(len(y_true))
        '''
        =============================================================================================================
        Save inference images to folder
        ==============================================================================================================
        '''
        
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        pred_label_dir = pred_dir +"/" +prediction
        if not os.path.exists(pred_label_dir):
            os.makedirs(pred_label_dir)
        """
        image name format: [GroundTruth]_[Predict]_[Score]_[img_count].png
        """
        save_pred_img_path = pred_label_dir + "/"+GT+"_"+prediction+"_" + str(score) + "_" + str(img_count)+".jpg"
        cv2.imwrite(save_pred_img_path,img)
        img_count = img_count + 1
        
        
'''
===================================================================
function : Analysis_Image_Path
Analysis inference image by image name
 image name format: [GroundTruth]_[Predict]_[Score]_[img_count].png
======================================================================
'''
def Analysis_Image_Path(img_path):
    #GT = os.path.basename(os.path.dirname(img_path))
    img_name = os.path.basename(img_path)
    GT = img_name.split("_")[0]
    predict = img_name.split("_")[1]
    score = img_name.split("_")[2]
    
    return GT,predict,score


def Calculate_Inference_Accuracy(imagedir,class_list):
    search_img_dir = imagedir + "/**/*.jpg"
    img_list = glob.iglob(search_img_dir)
    print('img_list:',img_list)
   
    
    print("len(class_list) =",len(class_list) )
    '''
    =============================================================================
    initailize count Correct, count Wrong, Recall
    ==============================================================================
    '''
    count_correct = [0]*len(class_list)  
    count_wrong = [0]*len(class_list)  
    recall = [0]*len(class_list)  
    
    for img_path in img_list:
        #print(img_path)
        '''
        =====================================================================
        Get GT, predict by img file name
        =====================================================================
        '''
        GT,predict,score = Analysis_Image_Path(img_path)
        index_cnt = 0    
        '''
        ======================================================================
        if GT==predict:
            then count Correct ++
        else:
            then count Wrong ++
        ======================================================================
        '''
        for label in class_list:
            if GT == label:
                if predict == label:
                    count_correct[index_cnt] = count_correct[index_cnt] + 1
                else:
                    count_wrong[index_cnt] = count_wrong[index_cnt] + 1
            index_cnt  = index_cnt + 1
    '''
    ========================================================================
    initial total
    total = count correct + count wrong
    ========================================================================
    '''
    total = [0]*len(class_list)
    for i in range(len(class_list)):
        total[i] = int(count_correct[i]) + int(count_wrong[i])
    print("total = ",total)
    '''
    =========================================================================
    recall = correct / total
    =========================================================================
    '''
    for i in range(len(class_list)):
        recall[i] = float(count_correct[i])/ float(total[i])
        
    print("recall = ",recall)
    print("class_list = ",class_list)
    '''
    ==========================================================================
    print recall result
    ==========================================================================
    '''
    for i in range(len(class_list)):
        #print(class_list[i])
        print(class_list[i],"total=",total[i],
              "correct:",count_correct[i],"wrong:",count_wrong[i],"recall:", str(int(recall[i]*100)),"%")


if __name__=="__main__":
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    main(repo_dir)
    
    