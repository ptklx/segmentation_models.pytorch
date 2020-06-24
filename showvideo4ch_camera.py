# import argparse
import os
# from glob import glob
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
# import functools
# import random
import torch
import torch.backends.cudnn as cudnn
# import numpy as np
# from torch.utils.data import DataLoader
import  segmentation_models_pytorch as smp
import platform
import time
winNolinux = True


#tensorrt  linux
# sys.path.append("D:/pengt/code/inference/tensorrt_project/torch2trt")
# from torch2trt import torch2trt

#  endtensorrt


model_ENCODER = 'timm-efficientnet-b0'  #0.080s  #'resnet101'  #'resnet50'
# model_ENCODER = 'timm-efficientnet-b1'  #  #'resnet101'  #'resnet50'
# model_ENCODER = 'mobilenet_v2_a' #0.070s
# model_ENCODER = 'resnet18' #0.070s
# model_ENCODER ="dpn68"

if platform.system().lower() == 'windows':
    winNolinux =True
    print("windows")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    DATA_DIR = 'D:/pengt/data/mydata'
    #video_path = '//SmartGo-Nas/pentao/data/video/111'
    # video_path='//SmartGo-Nas/pentao/data/video/liveBroadcast'
    out_video_path = "//SmartGo-Nas/pentao/data/video/big_out_video_win"
    model_path = '//SmartGo-Nas/pentao/code/4channels/parameter/seg_pytorch/%s/best_model_52_0.936.pth'%model_ENCODER
    # model_path = 'D:/pengt/segmetation/4channels/parameter/seg_pytorch/resnet18_Unet/best_model_last.pth'
   
    video_path = '//SmartGo-Nas/pentao/data/video/dancing/111/zhandouyouyang.mp4'

elif platform.system().lower() == 'linux':
    winNolinux = False
    print("linux")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    DATA_DIR = '/mnt/pentao/data/mydata'
    #linux
    # video_path = '/mnt/pentao/data/video/111'
    # out_video_path = "/mnt/pentao/data/video/big_out_video"
    # # model_path = '//SmartGo-Nas/pentao/code/4channels/Unet4/parameter/%s/best_model_last_0.pth'%model_ENCODER
    # model_path = '/mnt/pentao/code/4channels/Unet4/parameter_last/%s/best_model_9.pth'%model_ENCODER

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
 

#9    10    11     12    13    14    15    16    20    24    25    30   
#   *32
#288  320  352     384   416   448   480   512  640    786    800   960

img_width = 640 #448#448    #384  448  #512 #640
img_height = 480  #320#320   #288  320  #384  #480

#accelerate
# mean = [0.5, 0.5, 0.5,0]  #0.5
# std = [0.225, 0.225, 0.225,0.225]

mean=[0.485, 0.456, 0.406, 0]
std=[0.229, 0.224, 0.225, 1]

mean = np.array(mean)
std = np.array(std)
maskvalue = 64    #four channel value
zeroflag = 0      #
############
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


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


class segmentationPredict(object):
    def __init__(self):
        # ENCODER = "timm-efficientnet-b1"#model_ENCODER
        ENCODER_WEIGHTS = None
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
        # self.mode_trt = torch2trt(self.model)

        cudnn.benchmark = True   #
        self.pre_mask = np.zeros((img_height,img_width))
    def run(self,image,zeroflag =1,index=0):
        height, width = image.shape[0:2]
        bgr, bbx= resize_padding(image, [img_width,img_height])
        #bgr = cv2.resize(image, (img_width,img_height))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        m = np.expand_dims(self.pre_mask, -1)
        img = np.concatenate((rgb, m), -1)
        img = img / 255
        img = img - mean
        img = img / std
        img = to_tensor(img)
        rgbm = torch.from_numpy(img).to(self.DEVICE).unsqueeze(0)
        # print(rgbd.shape)
        # self.model.module(rgbm)
        t1 = time.time()
        first_pr_mask = self.model.predict(rgbm)
        # pr_mask_trt = self.mode_trt.predict(rgbm)
        first_pr_mask = first_pr_mask.squeeze().cpu().numpy()
        pr_mask = first_pr_mask[bbx[1]:bbx[3], bbx[0]:bbx[2]]   #########
        t2 = time.time()
        print ("predict",t2-t1)
        # pr_mask = np.zeros((img_height,img_width))
        # print(pr_mask)

        img_new_seg = (pr_mask > 0.5).astype(np.uint8)

        img_new_next = (pr_mask >0.85).astype(np.uint8)
        # if index%20==0:
        #     self.pre_mask = img_new_seg*0   #
        # else:
        self.pre_mask = (first_pr_mask>0.5).astype(np.uint8)*maskvalue   #
        if zeroflag:
            self.pre_mask = first_pr_mask * 0
        # plt.imshow(pr_mask)
        # plt.show()
        resultout = cv2.resize(pr_mask, (width,height), interpolation = cv2.INTER_LINEAR )  #cv2.INTER_AREA cv2.INTER_LINEAR
        #resultout=(resultout>0.5).astype(np.uint8)
        return resultout,img_new_seg,img_new_next



def MaxMinNormalization(x):
    Min = x.min()
    Max = x.max()
    x = (x - Min) / (Max - Min)
    return x

def get_edge(mask_ori):
    if len(mask_ori.shape)==3:
        if mask_ori.shape[2]==1:
            mask = mask_ori[:,:,0].copy()
        else:
            mask = mask_ori[0].copy()
    else:
        mask = mask_ori.copy()
    # find countours: img must be binary  //注意内轮廓和外轮廓  canny
    myImg = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    ret, binary = cv2.threshold(np.uint8(mask)*255, 127, 255, cv2.THRESH_BINARY)
    countours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL
    '''
    cv2.drawContours(myImg, countours, -1, 1, 10)
    diff = mask + myImg
    diff[diff < 2] = 0
    diff[diff == 2] = 1
    return diff   
    '''
    cv2.drawContours(myImg, countours, -1, 1, 3)   #最好一个参数是线宽度
    return myImg

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


# import time
def main():

    pthpredict = segmentationPredict()
    #pthpredictnext = segmentationPredict()
    # os.makedirs(os.path.join('outputs', config['name']), exist_ok=True)
    img_back = cv2.imread('D:/pengt/segmetation/test_pic/0.jpg')
    with torch.no_grad():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
        # fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
        out_height = 384#1080 #540#384 #720
        out_width = 640#1920 #960#640
        outpath = "D:/pengt/data/webvideo/joiner_mask_test.mp4"  #out
        # out = cv2.VideoWriter(outpath,fourcc, 20.0, (1280,720),True)
        #out1 = cv2.VideoWriter(largeoutpath,fourcc, 20.0, (out_width,out_height))
        video_path = "D:/pengt/data/webvideo/joiner.mp4"  #in
        cap=cv2.VideoCapture(0)
        # cap=cv2.VideoCapture(video_path)
        index =0
    #   background = cv2.resize(img_back,(width,height))

        while (True):
            ret,frame=cap.read()  
            if ret ==False:
                break
            index+=1  
            # if index<1200: ##########
                # continue
            # if index>2000:
                # break
            t1 = time.time()
            #img_ori = cv2.resize(frame, (out_width,out_height), interpolation=cv2.INTER_LINEAR)
            img_ori = frame
            height, width,_ = img_ori.shape


            background = cv2.resize(img_back,(width,height))  ####


            alphargb, pred,img_new_next = pthpredict.run(img_ori,0,index=index)
            
            # cv2.imshow("pre0.5",(pred*200).astype(np.uint8))
            # cv2.imshow("pre0.85",(img_new_next*200).astype(np.uint8))

            #########add   edge
            if 0:
                cannyedge =  get_edge((alphargb > 0.5).astype(np.uint8))  ##提取边缘
                #grayedge = cv2.resize(cannyedge, (out_width,out_height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                grayedge = cannyedge.astype(np.float32)
                cv2.imshow("edge",grayedge)
                grayedge[grayedge>0]=0.6
                alphargb = alphargb-grayedge
                alphargb[alphargb<0] = 0
                alphargb = alphargb.astype(np.float32)
                # alphargb = cv2.cvtColor(alphargb, cv2.COLOR_GRAY2BGR)
                # 
            ########end
            if 1 :  #增加美感
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
                destmask = cv2.morphologyEx(alphargb,cv2.MORPH_RECT,kernel)
                destmask = cv2.GaussianBlur( destmask, (3, 3), 0, 0)
                alphargb = destmask


            ###########
            alphargb = cv2.cvtColor(alphargb, cv2.COLOR_GRAY2BGR)
            cv2.imshow("alphargb",alphargb)
            result = np.uint8(img_ori * alphargb + background * (1-alphargb))
            myImg = np.ones((height, width*2 + 20, 3)) * 255
            myImg[:, :width, :] = img_ori
            myImg[:, width+20:, :] = result

            # out.write(result.astype(np.uint8))
            cv2.imshow("savepic",result.astype(np.uint8))

            # resultImg= myImg.astype(np.uint8)

            #cv2.imshow("ori",img_ori)
            t2 = time.time()
            print ("alltime:",t2-t1)
            # cv2.imshow("test",resultImg)


            if cv2.waitKey(1)&0xff == ord('q'):
                break
            
            print("index:%s"%index)
    
        cap.release()
        # out.release()  ####          
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
