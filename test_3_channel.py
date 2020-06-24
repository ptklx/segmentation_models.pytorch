#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 14:47:22 2019

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


best_model = model

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
mean = np.array(mean)
std = np.array(std)


import cv2
capture = cv2.VideoCapture(0) 


#capture.set(cv2.CAP_PROP_FPS,30)
#fps = capture.get(cv2.CAP_PROP_FPS) 


 
import time
i=0
fps = 0
kkk=0
time_start=time.time()
while True:
    
    ret,img = capture.read()
   
    img = img[:,0:480,:]
    img = cv2.resize(img,(512,512))
    
   
    rbg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rbg = rbg / 255.
    rbg = rbg - mean
    rbg = rbg / std

    rbg = to_tensor(rbg)
    rbg = torch.from_numpy(rbg).to('cuda').unsqueeze(0)
#    image = sample['image']
    
    
    
#    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(rbg)
    pr_mask = pr_mask.squeeze().cpu().numpy()
    pr_mask = pr_mask.round()
#    print ('time:',time.time() - t1)
    
    image0 = img.copy()
    back = image0*0+128
    for k in range(3):        
        image0[:,:,k] = image0[:,:,k] * pr_mask +back[:,:,k]*(1-pr_mask)
        
   
#    image2 = cv2.resize(image0,(640,640))
#    image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
    cv2.imshow('result', image0)
 #   cv2.putText(out,fps,(40,80),3,1.2,(123,123,123))
    kkk += 1
    i +=1
    time_end = time.time()
    
    if time_end - time_start > 1:
        time_start = time_end
        fps = i 
        print ('FPS:',fps)
        i=0
    
    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
  
        break
    
    
    
    
    
        
    