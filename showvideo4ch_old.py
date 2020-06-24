import argparse
import os
from glob import glob
import numpy as np
import sys
sys.path.append("D:/pengt/segmetation/4channels/Unet4/segmentation_models.pytorch-master")
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#linux
# image_path = '/mnt/pentao/data/mydata/val'
# val_image_list = '/mnt/pentao/data/mydata/val/valmask.txt'
# model_path = '/mnt/pentao/code/pytorch-nested-unet/4chmodels/mydata/val_NestedUNet_woDS/model_ori.pth'
# tempyml= '/mnt/pentao/code/pytorch-nested-unet/4chmodels/mydata/val_NestedUNet_woDS/config.yml'
#win10
# video_path = '//SmartGo-Nas/pentao/data/video/dancing/111'
video_path='//SmartGo-Nas/pentao/data/video/liveBroadcast'


model_ENCODER =  'resnet152'  #'resnet50'

# model_path = '//SmartGo-Nas/pentao/code/4channels/Unet4/parameter/%s/best_model_21.pth'%model_ENCODER
model_path = '//SmartGo-Nas/pentao/code/4channels/Unet4/parameter_last/%s_best/best_model_5.pth'%model_ENCODER

###########

#%%
# capture = cv2.VideoCapture(0) 
#capture.set(cv2.CAP_PROP_FPS,30)
#fps = capture.get(cv2.CAP_PROP_FPS) 

def get_mask(rgb,m):
    rgbm = np.concatenate((rgb, m), -1)
    rgbm = rgbm / 255.
    mean = [0.485, 0.456, 0.406, 0]
    std = [0.229, 0.224, 0.225, 1]
    mean = np.array(mean)
    std = np.array(std)
    rgbm = rgbm - mean
    rgbm = rgbm / std

    rgbm = to_tensor(rgbm)
    rgbm = torch.from_numpy(rgbm).to(DEVICE).unsqueeze(0)
    # print(rgbd.shape)
    pr_mask = model.predict(rgbm)
    # with torch.no_grad():
    #    prediction = model.forward(rgbd)
    pr_mask = pr_mask.squeeze().cpu().numpy().round()
    # print(pr_mask)
    pr_mask = (pr_mask > 0).astype(np.uint8)
    return pr_mask
 

img_width = 640#1280，1024#640  ,512， 800 ，384
img_height = 640 #960#960 #480

#accelerate
# mean = [0.5, 0.5, 0.5,0]  #0.5
# std = [0.225, 0.225, 0.225,0.225]

mean=[0.485, 0.456, 0.406, 0]
std=[0.229, 0.224, 0.225, 1]

mean = np.array(mean)
std = np.array(std)
maskvalue = 128    #four channel value
zeroflag = 0      #
############
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')



class UnetPredict(object):
    def __init__(self):
        ENCODER = model_ENCODER
        ENCODER_WEIGHTS = None
        DEVICE = 'cuda'
        CLASSES = ['person']
        ACTIVATION = 'sigmoid'

        # create segmentation model with pretrained encoder
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )
        model = model.cuda()
        self.model = torch.load(model_path)
        cudnn.benchmark = True   #
        self.pre_mask = np.zeros((img_height,img_width))
   


def main():
    ENCODER = model_ENCODER
    ENCODER_WEIGHTS = None
    DEVICE = 'cuda'
    CLASSES = ['person']
    ACTIVATION = 'sigmoid'

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )
    model = model.cuda()
    model = torch.load(model_path)
    cudnn.benchmark = True   #
    pre_mask = np.zeros((img_height,img_width))
    
    # os.makedirs(os.path.join('outputs', config['name']), exist_ok=True)
    with torch.no_grad():
        ve_list = os.listdir(video_path)
        n_b = 0 
        for v_name in ve_list:
            if os.path.splitext(v_name)[1]!=".mp4":
                continue               
            videofi = os.path.join(video_path,v_name)
            #avisuffix = os.path.splitext(v_name)[0]+".avi"
            # avisuffix = v_name
            # outpath = os.path.join(out_video_path,avisuffix)
            # largeoutpath =  os.path.join(out_video_path,"large"+avisuffix)
            print(n_b,"###")
            # if n_b != 2 :      #select video
            #     n_b+=1
            #     continue
            n_b+=1
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
            out_height = img_height #540#384
            out_width = img_width #960#640

            # out = cv2.VideoWriter(outpath,fourcc, 20.0, (out_width,out_height))
            # out1 = cv2.VideoWriter(largeoutpath,fourcc, 20.0, (out_width,out_height))

            cap=cv2.VideoCapture(videofi)
            index =0
            while (True):
                ret,frame=cap.read()  
                index+=1
                startindex = 200
                if index <startindex:   
                    continue
                if index>startindex+1600:     # 6000
                    break
                print(index)       
                if ret == True:
                    if True:   #video
                        bgr = cv2.resize(frame, ( img_width,img_height))
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        m = np.expand_dims(pre_mask, -1)
                        img = np.concatenate((rgb, m), -1)
                        img = img / 255
                        img = img - mean
                        img = img / std
                        img = to_tensor(img)
                        # img1 = torch.from_numpy(img).unsqueeze(0).cuda(0)
                        
                        rgbm = torch.from_numpy(img).to(DEVICE).unsqueeze(0)
                        # print(rgbd.shape)
                        pr_mask = model.predict(rgbm)
                        pr_mask = pr_mask.squeeze().cpu().numpy()
                        # print(pr_mask)
                        pr_mask = (pr_mask > 0.5).astype(np.uint8)

                        img_new_seg = pr_mask
                        if index%20==0:
                            pre_mask = img_new_seg*0   #
                        else:
                            pre_mask = img_new_seg*maskvalue   #
                        if zeroflag:
                            pre_mask = img_new_seg * 0
                        # plt.imshow(result_out)
                        # plt.show()
                        cv2.imshow("mask",img_new_seg*200)
                        cv2.imshow("bgr",bgr)
                        cv2.waitKey(1)
                        print("index%s"%index)
                       
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
