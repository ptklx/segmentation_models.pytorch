#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:36:14 2019

@author: xingshi2
"""



import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as albu
import torch
import time


#best_model = torch.load('/media/xingshi2/data/matting/Unet3/vgg_yxl_best_model.pth')


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


ENCODER = 'se_resnet50'
#ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
preprocessing=get_preprocessing(preprocessing_fn)



def get_iou(mask,resized_gt):
    mask[mask>0] = 1
    gt[gt>0] = 1
    tmp = mask + resized_gt
    
    num = tmp==2
    den = tmp > 0
    iou = np.sum(num) / np.sum(den)
    return iou



dirr = '/media/xingshi2/data/matting/mobile-deeplab-v3-plus/datasets/people_segmentation/'
data_type = 'val'
names = dirr + 'segmentation/' + data_type  + '.txt'

f = open(names)
names = f.read()
f.close()

names = names.split()

input_size = 512

iou_all = []
time_start=time.time()
filp = False

for i in range(len(names)):
    name_tmp = dirr + 'images/' + names[i] + '.jpg'
    mask_tmp = dirr + 'masks/' + names[i] + '.png'
    
    img = cv2.imread(name_tmp)
    gt = cv2.imread(mask_tmp)[:,:,0]
    
    h,w = np.shape(gt)
    l = np.max((h,w))
    img1 = np.zeros((l,l,3)).astype('uint8')
    img1[:h,:w,:] = img
    img1 = cv2.resize(img1,(input_size,input_size))
   
    
    img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    sample = preprocessing(image=img2)
    image = sample['image']
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    
    if not filp:
         pr_mask = best_model.predict(x_tensor)
         mask = pr_mask.squeeze().cpu().numpy().round().astype('uint8')
    else:
        img1 = cv2.flip(img,1,dst=None)
        sample1 = preprocessing(image=img1)
        image1 = sample1['image']
        x_tensor1 = torch.from_numpy(image1).to(DEVICE).unsqueeze(0)
        
        x = torch.cat((x_tensor,x_tensor1),0)
        pr_mask = best_model.predict(x)
        mask_tmp = pr_mask.squeeze().cpu().numpy()
        mask0 = mask_tmp[0]
        mask1 = cv2.flip(mask_tmp[1],1,dst=None)
        mask = ((mask0+mask1)/2).round().astype('uint8')
    
    mask = cv2.resize(mask,(l,l))[:h,:w]
   
    
    iou = get_iou(mask,gt)
    iou_all.append(iou)
    
   
    out_img = img.copy()
    for k in range(3):        
        out_img[:,:,k] = out_img[:,:,k] * mask
    out_img=out_img[...,::-1]
#    if iou < 0.8:
    oo = np.zeros((h,w*2,3)).astype('uint8')
    oo[:h,:w,:] = img[...,::-1]
    oo[:h,w:2*w,:] = out_img
#        cv2.imwrite('/media/xingshi/新加卷/数据库/myparsing_data/bbbb/' + names[i] + '.jpg',out_img)
    cv2.imwrite('/media/xingshi2/data/tmp_data/' + str(iou)+'_' + names[i] + '.jpg',oo)

    print(i)



fps = 1000/(time.time() - time_start)

avg_iou = np.mean(iou_all)
print (avg_iou)
print (fps)
    

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
sample = preprocessing(image=img)
image = sample['image']
x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

time_start=time.time()
for i in range(100):
    pr_mask = best_model.predict(x_tensor)
fps = 100/(time.time() - time_start)         
print('no filp:',fps)    
    
    
x_tensor = torch.cat((x_tensor,x_tensor),0)    
time_start=time.time()
for i in range(1000):
    pr_mask = best_model.predict(x_tensor)
fps = 1000/(time.time() - time_start)         
print('no filp:',fps) 




from torch.autograd import Variable
import onnx
dummy_input = Variable(torch.randn(1, 3, 640, 480)).cpu()
model=model.cpu()
torch.onnx.export(model, dummy_input, "/media/xingshi2/data/purning/network-slimming/Unet_vgg.onnx", verbose=True)

model1 = onnx.load("/media/xingshi2/data/purning/network-slimming/Unet_resnet18asd.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model1)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model1.graph)
opset_version = model1.opset_import[0].version





#%%
import torch
import sys
sys.path.append('/home/mydisk/code/matting/Unet4/segmentation_models.pytorch-master')
import segmentation_models_pytorch as smp
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# ENCODER = 'se_resnext50_32x4d'
ENCODER = 'resnet50'
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

# resume
model_path='/home/pt/Desktop/code/matting/traindata/unet4/best_model.pth'
model = torch.load(model_path)
model = model.cuda()


#%%
import time
import cv2
import numpy as np

capture = cv2.VideoCapture(0) 


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
 

done=False
i = 0
start_time=time.time()
pre_mask=np.zeros((480, 640))
png=0

mean = [0.485, 0.456, 0.406, 0]
std = [0.229, 0.224, 0.225, 1]
mean = np.array(mean)
std = np.array(std)
kernel = np.ones((5, 5), np.uint8)

while True:

    start = time.time()
    sucess, img = capture.read()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   
    
    start = time.time()
    if i%40==0:
      
        mm = rgb[:,:,0:1] * 0
        mask = get_mask(rgb,mm)
        mask = mask.astype(np.uint8)
    else:
        mm = pre_mask*128
        mm = np.expand_dims(mm, -1)
        mask = get_mask(rgb,mm)
        mask = mask.astype(np.uint8)
# use dilation
#    pre_mask = cv2.dilate(mask,kernel,iterations = 1)
       

    pre_mask = mask.copy()    
    for k in range(3):
            # img[:,:,k] = img[:,:,k] * mask + (1-mask)*back[:,:,k]
        img[:, :, k] = img[:, :, k] * mask

    cv2.imshow('pre_mask', mm)
    cv2.imshow('result', img)
    i+=1     

    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()

        break

