import argparse
import os
from glob import glob
import numpy as np
import sys
sys.path.append("D:/pengt/segmetation/4channels/Unet4/segmentation_models.pytorch-master")
sys.path.append("/mnt/pentao/code/4channels/Unet4/segmentation_models.pytorch-master")
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

model_ENCODER =  "resnet18" #'resnet50'  #'resnet50'

#win10
video_path = '//SmartGo-Nas/pentao/data/video/dancing/111/zhandouyouyang.mp4'
out_video_path = "//SmartGo-Nas/pentao/data/video/big_out_video"
# model_path = '//SmartGo-Nas/pentao/code/4channels/Unet4/parameter/%s/best_model_last_0.pth'%model_ENCODER
model_path = '//SmartGo-Nas/pentao/code/4channels/Unet4/parameter_last/%s/best_model_9.pth'%model_ENCODER

# #linux
# video_path = '/mnt/pentao/data/video/111'
# out_video_path = "/mnt/pentao/data/video/big_out_video"
# # model_path = '//SmartGo-Nas/pentao/code/4channels/Unet4/parameter/%s/best_model_last_0.pth'%model_ENCODER
# model_path = '/mnt/pentao/code/4channels/Unet4/parameter_last/%s/best_model_9.pth'%model_ENCODER



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
 

img_width = 1024#1280，1024#640  ,512， 800 ，384
img_height = 960#960 #480

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
        self.DEVICE = 'cuda'
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
    def run(self,image,zeroflag =1,index=0):
        height, width = image.shape[0:2]
        bgr = cv2.resize(image, (img_width,img_height))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        m = np.expand_dims(self.pre_mask, -1)
        img = np.concatenate((rgb, m), -1)
        img = img / 255
        img = img - mean
        img = img / std
        img = to_tensor(img)
        rgbm = torch.from_numpy(img).to(self.DEVICE).unsqueeze(0)
        # print(rgbd.shape)
        pr_mask = self.model.predict(rgbm)
        pr_mask = pr_mask.squeeze().cpu().numpy()
        # print(pr_mask)
        img_new_seg = (pr_mask > 0.5).astype(np.uint8)

        
        if index%20==0:
            self.pre_mask = img_new_seg*0   #
        else:
            self.pre_mask = img_new_seg*maskvalue   #
        if zeroflag:
            self.pre_mask = img_new_seg * 0

        resultout = cv2.resize(pr_mask, (width,height), interpolation = cv2.INTER_LINEAR )  #cv2.INTER_AREA cv2.INTER_LINEAR
        resultout=(resultout>0.5).astype(np.uint8)
        return resultout

def resize_padding(image, dstshape, padValue=0):    # 等比例补边
    height, width, _ = image.shape
    ratio = float(width)/height # ratio = (width:height)
    dst_width = int(min(dstshape[1]*ratio, dstshape[0]))
    dst_height = int(min(dstshape[0]/ratio, dstshape[1]))
    origin = [int((dstshape[1] - dst_height)/2), int((dstshape[0] - dst_width)/2)]
    if len(image.shape)==3:
        image_resize = cv2.resize(image, (dst_width, dst_height))
        newimage = np.zeros(shape = (dstshape[1], dstshape[0], image.shape[2]), dtype = np.uint8) + padValue
        newimage[origin[0]:origin[0]+dst_height, origin[1]:origin[1]+dst_width, :] = image_resize
        bbx = [origin[1], origin[0], origin[1]+dst_width, origin[0]+dst_height] # x1,y1,x2,y2
    else:
        image_resize = cv2.resize(image, (dst_width, dst_height),  interpolation = cv2.INTER_NEAREST)
        newimage = np.zeros(shape = (dstshape[1], dstshape[0]), dtype = np.uint8)
        newimage[origin[0]:origin[0]+height, origin[1]:origin[1]+width] = image_resize
        bbx = [origin[1], origin[0], origin[1]+dst_width, origin[0]+dst_height] # x1,y1,x2,y2
    return newimage, bbx

def MaxMinNormalization(x):
    Min = x.min()
    Max = x.max()
    x = (x - Min) / (Max - Min)
    return x



def main():

    pthpredict = UnetPredict()
    pthpredictnext = UnetPredict()
    # os.makedirs(os.path.join('outputs', config['name']), exist_ok=True)



    with torch.no_grad():
        ve_list = os.listdir(video_path)
        n_b = 0 
        for v_name in ve_list:               
            videofi = os.path.join(video_path,v_name)
            #avisuffix = os.path.splitext(v_name)[0]+".avi"
            avisuffix = v_name
            outpath = os.path.join(out_video_path,avisuffix)
            largeoutpath =  os.path.join(out_video_path,"large"+avisuffix)
            print(n_b,"###")

            if n_b != 1 :      #select video
                n_b+=1
                continue

            n_b+=1
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
            out_height = 1080 #540#384
            out_width = 1920 #960#640

            out = cv2.VideoWriter(outpath,fourcc, 20.0, (out_width,out_height))
            out1 = cv2.VideoWriter(largeoutpath,fourcc, 20.0, (out_width,out_height))

            cap=cv2.VideoCapture(videofi)
            index =0
            while (True):
                ret,frame=cap.read()  
                index+=1
                startindex = 5000
                if index <startindex:   
                    continue
                if index>startindex+6000:     # 6000
                    break
                print(index)       
                if ret == True:
                    img_orig = cv2.resize(frame, (out_width,out_height), interpolation=cv2.INTER_LINEAR)
                    bg_mask = pthpredict.run(img_orig,0,index=index)
                    if bg_mask.sum()>5*5:
                        index_t = np.argwhere(bg_mask > 0)
                        row_h= [index_t[:,0].min(),index_t[:,0].max()]
                        col_w= [index_t[:,1].min(),index_t[:,1].max()]
                    
                        ori_row_h= np.dot(row_h,(frame.shape[0]/out_height)).astype(np.int32)
                        ori_col_w= np.dot(col_w,(frame.shape[1]/out_width)).astype(np.int32)

                        ori_h ,ori_w ,_=frame.shape
                        gap_height = int((ori_row_h[1]-ori_row_h[0])/7)
                        gap_width = int((ori_col_w[1]-ori_col_w[0])/4)
                        dest_row_h=[0,ori_h]
                        dest_col_w=[0,ori_w]
                        dest_row_h[0] = 0 if (ori_row_h[0]-gap_height)<0 else ori_row_h[0]-gap_height
                        dest_row_h[1] = ori_h if (ori_row_h[1]+gap_height)>ori_h else ori_row_h[1]+gap_height
                        dest_col_w[0] = 0 if (ori_col_w[0]-gap_width)<0 else ori_col_w[0]-gap_width
                        dest_col_w[1] = ori_w if (ori_col_w[1]+gap_width)>ori_w else ori_col_w[1]+gap_width

                        # newimg = frame[ori_row_h[0]:ori_row_h[1],
                        #              ori_col_w[0]:ori_col_w[1]]  # 裁剪坐标为[y0:y1, x0:x1]   人体
                        
                        newimg = frame[dest_row_h[0]:dest_row_h[1],
                                        dest_col_w[0]:dest_col_w[1]]  # 裁剪坐标为[y0:y1, x0:x1]   人体


                        #whole_man = cv2.resize(newimg, (out_width,out_height), interpolation=cv2.INTER_LINEAR)
                        in_shape = newimg.shape
                        in_img, bbx= resize_padding(newimg, [out_width,out_height])
                        whole_bg_mask = pthpredictnext.run(in_img,0,index=index)
                        out_get = whole_bg_mask[bbx[1]:bbx[3], bbx[0]:bbx[2]]
                        out_get = cv2.resize(out_get, (in_shape[1], in_shape[0]))

                        #tempf = np.copy( frame)
                        big_f =np.zeros((frame.shape[0],frame.shape[1]), dtype = np.uint8) 

                        # big_f[ori_row_h[0]:ori_row_h[1],ori_col_w[0]:ori_col_w[1]] [:] = out_get
                        big_f[dest_row_h[0]:dest_row_h[1],dest_col_w[0]:dest_col_w[1]] [:] = out_get
                        # temp=temp+whole_bg_mask
                        #seg_imgnext = np.copy(frame)
                        kernel = np.ones((3,3),np.uint8)  
                        big_f = cv2.erode(big_f,kernel,iterations = 1)
                        if True:
                            seg_imgnext=np.zeros((frame.shape[0],frame.shape[1],frame.shape[2]), dtype = np.uint8) 
                            seg_imgnext[:, :, 0] = frame[:, :, 0] * big_f 
                            seg_imgnext[:, :, 1] = frame[:, :, 1] * big_f 
                            seg_imgnext[:, :, 2] = frame[:, :, 2] * big_f 
                            # cv2.imshow("test",seg_imgnext)
                            out.write(seg_imgnext)
                       
                        # erosion = cv2.erode(big_f,kernel,iterations = 1)
                        kernel1 = np.ones((5,5),np.uint8) 
                        erosion = cv2.erode(big_f,kernel,iterations = 1)

                        blur = cv2.blur(big_f*255,(5,5))
                        normal = MaxMinNormalization(blur)
                        next_f = normal
                        seg_imgnext_er=np.zeros((frame.shape[0],frame.shape[1],frame.shape[2]), dtype = np.float64) 
                        seg_imgnext_er[:, :, 0] = frame[:, :, 0] * next_f 
                        seg_imgnext_er[:, :, 1] = frame[:, :, 1] * next_f 
                        seg_imgnext_er[:, :, 2] = frame[:, :, 2] * next_f 
                        seg_imgnext_er=seg_imgnext_er.astype(np.uint8)

                        # tempfile = '//SmartGo-Nas/pentao/data/video/small_pic/ori.jpg'
                        # tempfilemask = '//SmartGo-Nas/pentao/data/video/small_pic/mask.jpg'
                        # cv2.imwrite(filename, frame)
                        # cv2.imwrite(tempfilemask, seg_imgnext_er)

                        # cv2.imshow("erosion",seg_imgnext_er)
                        # cv2.waitKey(0)
                        #cv2.imshow("mask",whole_bg_mask*200)
                        
                        #cv2.imshow("test",newimg)
                        #cv2.waitKey(1)

                        # seg_out_next = cv2.resize(seg_imgnext, (out_width,out_height), interpolation= cv2.INTER_AREA ) #cv2.INTER_AREA   cv2.INTER_LINEAR
                        # cv2.imshow("big",seg_out_next)
                        # cv2.waitKey(1)

                        out1.write(seg_imgnext_er)


                    print("index%s"%index)
            cap.release()          
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
