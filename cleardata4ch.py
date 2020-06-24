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

# import segmentation_models_pytorch as smp

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import platform

# sys.path.append("D:/pengt/segmetation/4channels/Unet4/segmentation_models.pytorch-master")
winNolinux = True
if platform.system().lower() == 'windows':
    winNolinux =True
    print("windows")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # image_path = 'D:\\pengt\\data\\mydata\\train\\images'  #jpg
    # mask_path = 'D:\\pengt\\data\\mydata\\train\\masks'  #png
    # out_path = 'D:\\pengt\\data\\my_cleardata'
    # image_path = '//SmartGo-Nas\\pentao\\data\sinet_extremenet\\Baidu_seg\\baidu_V1\\input'  #png
    # mask_path = '//SmartGo-Nas\\pentao\\data\\sinet_extremenet\\Baidu_seg\\baidu_V1\\target'  #png
    # out_path = '//SmartGo-Nas\\pentao\\data\\my_cleardata2\\baidu_v1'

    # image_path = '//SmartGo-Nas/pentao/data/matting_human_haf/clip_img'  #jpg
    # mask_path = '//SmartGo-Nas/pentao/data/matting_human_haf/matting'   # png
    # out_path = '//SmartGo-Nas/pentao/data/my_cleardata2/matting_haf'

    image_path = '//SmartGo-Nas/pentao/data/humanparsing/JPEGImages'  #jpg
    mask_path = '//SmartGo-Nas/pentao/data/humanparsing/SegmentationClassAug'   # jpg
    out_path = '//SmartGo-Nas/pentao/data/my_cleardata/humanparsing'


elif platform.system().lower() == 'linux':
    winNolinux = False
    print("linux")
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # image_path = '/mnt/pentao/data/mydata/train/images'
    # mask_path = '/mnt/pentao/data/mydata/train/masks'
    # out_path = '/mnt/pentao/data/my_cleardata'
    # image_path = '/mnt/pentao/data/sinet_extremenet/EG1800/Images'  #png
    # mask_path = '/mnt/pentao/data/sinet_extremenet/EG1800/Labels'   # png
    # out_path = '/mnt/pentao/data/my_cleardata2/eg1800'

    image_path = '/mnt/pentao/data/matting_human_haf/clip_img'  #jpg
    mask_path = '/mnt/pentao/data/matting_human_haf/matting'   # png
    out_path = '/mnt/pentao/data/my_cleardata2/matting_haf'

mattinghumanhaf = False   #False


model_ENCODER =  'timm-efficientnet-b0'  #'resnet50'


if winNolinux:
    model_path = '//SmartGo-Nas/pentao/code/4channels/parameter/seg_pytorch/%s/best_model_13_0.927.pth'%model_ENCODER
else:
    model_path = '/mnt/pentao/code/4channels/parameter/seg_pytorch/%s/best_model_13_0.927.pth'%model_ENCODER



save_pathlist = ['negSample','little','big','bighard','other']
save_classes = ['images','masks']

for fi_p in save_pathlist:
    for fi_l in save_classes:
        os.makedirs(os.path.join(out_path,fi_p,fi_l), exist_ok=True)
#%%

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
img_height = 640#960 #480

#accelerate
# mean = [0.5, 0.5, 0.5,0]  #0.5
# std = [0.225, 0.225, 0.225,0.225]

mean=[0.485, 0.456, 0.406, 0]
std=[0.229, 0.224, 0.225, 1]

mean = np.array(mean)
std = np.array(std)
maskvalue = 128    #four channel value
zeroflag = 1     #
############
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')



class UnetPredict(object):
    def __init__(self):
        # ENCODER = model_ENCODER
        # ENCODER_WEIGHTS = None
        self.DEVICE = 'cuda'
        # CLASSES = ['person']
        # ACTIVATION = 'sigmoid'

        # create segmentation model with pretrained encoder
        # model = smp.Unet(
        #     encoder_name=ENCODER,
        #     encoder_weights=ENCODER_WEIGHTS,
        #     classes=len(CLASSES),
        #     activation=ACTIVATION,
        # )
        # model = model.cuda()
        self.model = torch.load(model_path)
        self.model = self.model.cuda()
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

        img_new_next = None #(pr_mask > 0.85).astype(np.uint8)
        if index%20==0:
            self.pre_mask = img_new_seg*0   #
        else:
            self.pre_mask = img_new_seg*maskvalue   #
        if zeroflag:
            self.pre_mask = img_new_seg * 0
        #plt.imshow(pr_mask)
        #plt.show()
        resultout = cv2.resize(pr_mask, (width,height), interpolation = cv2.INTER_LINEAR )  #cv2.INTER_AREA cv2.INTER_LINEAR
        img_new_seg =  cv2.resize(img_new_seg, (width,height), interpolation = cv2.INTER_LINEAR ) 
        #resultout=(resultout>0.5).astype(np.uint8)
        return resultout,img_new_seg,img_new_next

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

def get_iou(mask,resized_gt):
        mask[mask>0] = 1
        resized_gt[resized_gt>0] = 1
        tmp = mask + resized_gt
        num = tmp==2
        den = tmp > 0
        if np.sum(den)==0:
            return 1
        iou = np.sum(num) / np.sum(den)
        return iou

    # import time
def main():

    pthpredict = UnetPredict()
    #pthpredictnext = UnetPredict()
    # os.makedirs(os.path.join('outputs', config['name']), exist_ok=True)
    rightNum = 0
    with torch.no_grad():
        if mattinghumanhaf:
            needswap = ["clip","matting"]
            img_folds_0 = os.listdir(image_path)
            for sub0 in img_folds_0:
                mask_fold_0 = os.path.join(mask_path,sub0)
                img_fold_0 = os.path.join(image_path,sub0)
                img_folds_1 = os.listdir(img_fold_0)
                for sub1 in img_folds_1:
                    img_ids = glob(os.path.join(img_fold_0,sub1, '*.jpg'))
                    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
                    for index,ids in enumerate(img_ids):
                        # if index<4934:
                        #     continue
                        img = cv2.imread(os.path.join(img_fold_0,sub1,ids + ".jpg"))
                        mask_rgb= cv2.imread(os.path.join(mask_fold_0,os.path.split(sub1)[1].replace(needswap[0],needswap[1]),
                            ids + ".png"),cv2.IMREAD_UNCHANGED)
                        if mask_rgb is None:
                            continue
                        mask = mask_rgb[:,:,3]
                        mask = mask[:,:,None]
                        # mask[mask > 125] = 1
                        # mask[mask < 126] = 0

                        mask = np.where(mask < 125, 0, 1)
                        # mask[mask > 0] = 1
                        mask = mask.astype(np.uint8)   
                        img_ori = img
                        height, width,_ = img_ori.shape
                        
                        ori_point = np.sum(mask)
                        if ori_point==0:
                            cv2.imwrite(os.path.join(out_path,save_pathlist[0],save_classes[0],ids + ".jpg"),img)
                            cv2.imwrite(os.path.join(out_path,save_pathlist[0],save_classes[1],ids + ".png"),mask*200)
                            continue
                        all_pint = height*width
                        ratio_m =  ori_point/all_pint
                        contours,hierarchy = cv2.findContours(mask*255,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                        contourNum = len(contours)
                        def cnt_area(cnt):
                            area = cv2.contourArea(cnt)
                            return area
                        
                        contours.sort(key = cnt_area, reverse=True)  # reverse=False 升
                        #1/100    1/20
                        if cv2.contourArea(contours[0])/all_pint<0.01 and ratio_m<0.05:

                            cv2.imwrite(os.path.join(out_path,save_pathlist[1],save_classes[0],ids + ".jpg"),img)
                            cv2.imwrite(os.path.join(out_path,save_pathlist[1],save_classes[1],ids + ".png"),mask*200)
                            # cv2.imshow("test",mask*200) 
                            # cv2.waitKey(0)
                            continue

                        alphargb, pred,img_new_next = pthpredict.run(img_ori,0,index=index)
                        iou = get_iou(mask,pred[:,:,None])
                        
                        pre_point = np.sum(pred)  ####
                    
                        if iou>0.90:
                            cv2.imwrite(os.path.join(out_path,save_pathlist[2],save_classes[0],ids + ".jpg"),img)
                            cv2.imwrite(os.path.join(out_path,save_pathlist[2],save_classes[1],ids + ".png"),pred[:,:,None]*200)
                            rightNum+=1

                        elif iou>0.5:
                            cv2.imwrite(os.path.join(out_path,save_pathlist[2],save_classes[0],ids + ".jpg"),img)
                            cv2.imwrite(os.path.join(out_path,save_pathlist[2],save_classes[1],ids + ".png"),mask*200)
                        elif iou>0.4 and ratio_m>0.4 and cv2.contourArea(contours[0])/all_pint>0.4:
                            cv2.imwrite(os.path.join(out_path,save_pathlist[3],save_classes[0],ids + ".jpg"),img)
                            cv2.imwrite(os.path.join(out_path,save_pathlist[3],save_classes[1],ids + ".png"),mask*200)
                        else:
                            cv2.imwrite(os.path.join(out_path,save_pathlist[4],save_classes[0],ids + ".jpg"),img)
                            cv2.imwrite(os.path.join(out_path,save_pathlist[4],save_classes[1],ids + ".png"),mask*200)
                            print("iou:",iou)
                            # cv2.imshow("ori_img",img) 
                            # cv2.imshow("ori_mask",mask*200)
                            # cv2.imshow("pre0.5",(pred*200).astype(np.uint8))
                            # cv2.imshow("combine",(pred[:,:,None]*100+mask*255).astype(np.uint8))
                            # cv2.waitKey(0)

                        print("right:",rightNum,"allnum:",index,"img_ids:",ids)
                        # save_pathlist = ['negSample','little','big','other']
                        # save_classes = ['images','masks']
                        
                        # img_path = os.path.join(img_dir,img)
                        # cv2.imwrite(os.path.join(dst_dir,img_name),image)

                        # cv2.imshow("ori_img",img) 
                        # cv2.imshow("ori_mask",mask*200)
                        # cv2.imshow("pre0.5",(pred*200).astype(np.uint8))
                        # cv2.imshow("combine",(pred[:,:,None]*100+mask*255).astype(np.uint8))
                        # cv2.waitKey(0)
                

        else:        
            img_ids = glob(os.path.join(image_path, '*.jpg'))
            img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
            for index,ids in enumerate(img_ids):
                # if index<4934:
                #     continue
                img = cv2.imread(os.path.join(image_path,ids + ".jpg"))
                mask= cv2.imread(os.path.join(mask_path,ids + ".png"), cv2.IMREAD_GRAYSCALE)[..., None]
                # mask= cv2.imread(os.path.join(mask_path,ids + "-profile.jpg"), cv2.IMREAD_GRAYSCALE)[..., None]
                if mask is None:
                    continue
                # mask[mask > 125] = 1
                # mask[mask < 126] = 0

                #mask = np.where(mask < 125, 1, 0)     #mask = np.where(mask < 125, 0, 1)  
                mask[mask > 0] = 1
                mask = mask.astype(np.uint8)   
                img_ori = img
                height, width,_ = img_ori.shape
                
                ori_point = np.sum(mask)
                if ori_point==0:
                    cv2.imwrite(os.path.join(out_path,save_pathlist[0],save_classes[0],ids + ".jpg"),img)
                    cv2.imwrite(os.path.join(out_path,save_pathlist[0],save_classes[1],ids + ".png"),mask*200)
                    continue
                all_pint = height*width
                ratio_m =  ori_point/all_pint
                contours,hierarchy = cv2.findContours(mask*255,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                contourNum = len(contours)
                def cnt_area(cnt):
                    area = cv2.contourArea(cnt)
                    return area
                
                contours.sort(key = cnt_area, reverse=True)  # reverse=False 升
                #1/100    1/20
                if cv2.contourArea(contours[0])/all_pint<0.01 and ratio_m<0.05:

                    cv2.imwrite(os.path.join(out_path,save_pathlist[1],save_classes[0],ids + ".jpg"),img)
                    cv2.imwrite(os.path.join(out_path,save_pathlist[1],save_classes[1],ids + ".png"),mask*200)
                    # cv2.imshow("test",mask*200) 
                    # cv2.waitKey(0)
                    continue

                alphargb, pred,img_new_next = pthpredict.run(img_ori,0,index=index)
                iou = get_iou(mask,pred[:,:,None])
                
                pre_point = np.sum(pred)  ####
            
                if iou>0.90:
                    cv2.imwrite(os.path.join(out_path,save_pathlist[2],save_classes[0],ids + ".jpg"),img)
                    cv2.imwrite(os.path.join(out_path,save_pathlist[2],save_classes[1],ids + ".png"),pred[:,:,None]*200)
                    rightNum+=1

                elif iou>0.5:
                    cv2.imwrite(os.path.join(out_path,save_pathlist[2],save_classes[0],ids + ".jpg"),img)
                    cv2.imwrite(os.path.join(out_path,save_pathlist[2],save_classes[1],ids + ".png"),mask*200)
                elif iou>0.4 and ratio_m>0.4 and cv2.contourArea(contours[0])/all_pint>0.4:
                    cv2.imwrite(os.path.join(out_path,save_pathlist[3],save_classes[0],ids + ".jpg"),img)
                    cv2.imwrite(os.path.join(out_path,save_pathlist[3],save_classes[1],ids + ".png"),mask*200)
                else:
                    cv2.imwrite(os.path.join(out_path,save_pathlist[4],save_classes[0],ids + ".jpg"),img)
                    cv2.imwrite(os.path.join(out_path,save_pathlist[4],save_classes[1],ids + ".png"),mask*200)
                    print("iou:",iou)
                    # cv2.imshow("ori_img",img) 
                    # cv2.imshow("ori_mask",mask*200)
                    # cv2.imshow("pre0.5",(pred*200).astype(np.uint8))
                    # cv2.imshow("combine",(pred[:,:,None]*100+mask*255).astype(np.uint8))
                    # cv2.waitKey(0)

                print("right:",rightNum,"allnum:",index,"img_ids:",ids)
                # save_pathlist = ['negSample','little','big','other']
                # save_classes = ['images','masks']
                
                # img_path = os.path.join(img_dir,img)
                # cv2.imwrite(os.path.join(dst_dir,img_name),image)

                # cv2.imshow("ori_img",img) 
                # cv2.imshow("ori_mask",mask*200)
                # cv2.imshow("pre0.5",(pred*200).astype(np.uint8))
                # cv2.imshow("combine",(pred[:,:,None]*100+mask*255).astype(np.uint8))
                # cv2.waitKey(0)

            
         
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
