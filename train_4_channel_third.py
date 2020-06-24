#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
# import cv2
# import matplotlib.pyplot as plt
import functools
import random
import torch
# import numpy as np
from torch.utils.data import DataLoader
import  segmentation_models_pytorch as smp
from mydatasets import MyDataset
from mydatasets_select import MyDataset_select
import platform


# sys.path.append("D:/pengt/segmetation/4channels/Unet4/segmentation_models.pytorch-master")
winNolinux = True
if platform.system().lower() == 'windows':
    winNolinux =True
    print("windows")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    DATA_DIR = 'D:/pengt/data/mydata'
elif platform.system().lower() == 'linux':
    winNolinux = False
    print("linux")
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    DATA_DIR = '/mnt/pentao/data/mydata'
    #DATA_DIR = '/mnt/pentao/data/my_cleardata'
    #DATA_DIR = '/root/pengtao/data/mydata'
#save_path ='D:/pengt/segmetation/4channels/Unet4/parameter/best_model.pth'
#DATA_DIR = os.getcwd() + '/COCO/'

# x_train_dir = os.path.join(DATA_DIR, 'train/images')
# y_train_dir = os.path.join(DATA_DIR, 'train/masks')

# x_train_dir = DATA_DIR
# y_train_dir = DATA_DIR
x_train_dir = os.path.join(DATA_DIR, 'zjl_clear/images')
y_train_dir = os.path.join(DATA_DIR, 'zjl_clear/masks')

x_valid_dir = os.path.join(DATA_DIR, 'val/images')   #test/val/iamges
y_valid_dir = os.path.join(DATA_DIR, 'val/masks')

# y_valid_pre_dir = os.path.join(DATA_DIR, 'val/pre_masks')

x_test_dir = os.path.join(DATA_DIR, 'test/test/images')
y_test_dir = os.path.join(DATA_DIR, 'test/test/masks')

if winNolinux==True:
    x_train_dir = x_test_dir
    y_train_dir = y_test_dir
    x_valid_dir = x_test_dir
    y_valid_dir = y_test_dir

y_valid_pre_dir=None
y_train_pre_dir=None


# ENCODER 

#0-10M
#vgg13 vgg13_bn  vgg11_bn  vgg11  vgg16 vgg16_bn
#densenet121  
# efficientnet-b0  efficientnet-b1  efficientnet-b2  efficientnet-b3
#mobilenet_v2  


#10-20M
#'resnet18'  'dpn68'  'dpn68b' 'vgg16' #'vgg19'   densenet169  densenet201  efficientnet-b4
#timm-efficientnet-b3

ENCODER=  'timm-efficientnet-b0' #'dpn68'  #'timm-efficientnet-b0'  #"mobilenet_v2"   'resnet18'# 
# ENCODER= 'timm-efficientnet-b0'  #"mobilenet_v2"  
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['person']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'
use4channal =True


if winNolinux:
    x_train_dir = x_test_dir
    y_train_dir = y_test_dir
    flagt = 'a'
    save_path = '../parameter/seg_pytorch/%s_%s/best_model'%(ENCODER,flagt)
    save_last_path = '../parameter/seg_pytorch/%s_%s/best_model_last.pth'%(ENCODER,flagt)
    pre_path = '../parameter/seg_pytorch/%s_%s/best_model.pth'%(ENCODER,flagt)
    init_pre_path=''
    #save_path ='D:/pengt/segmetation/4channels/Unet4/parameter/%s/best_model.pth'%ENCODER
else:
    flagt = 'b'
    save_path = '../parameter/seg_pytorch/%s_%s/best_model'%(ENCODER,flagt)
    save_last_path = '../parameter/seg_pytorch/%s_%s/best_model_last.pth'%(ENCODER,flagt)
    #pre_path = '../parameter/seg_pytorch/%s/best_model.pth'%ENCODER
    #save_path ='D:/pengt/segmetation/4channels/Unet4/parameter/%s/best_model.pth'%ENCODER
    init_pre_path='../parameter/seg_pytorch/%s/best_model_5_0.9229.pth'%ENCODER

folder_path, file_name = os.path.split(save_path)
# if is not ospath.exists(folder_path)
os.makedirs(folder_path, exist_ok=True)



# model = smp.FPN(
#     encoder_name=ENCODER, 
#     encoder_weights=ENCODER_WEIGHTS, 
#     classes=len(CLASSES), 
#     activation=ACTIVATION,
# )

model = smp.DeepLabV3Plus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
    in_channels = 4,
)

# model = smp.Unet(
#     encoder_name=ENCODER, 
#     encoder_weights=ENCODER_WEIGHTS, 
#     classes=len(CLASSES), 
#     activation=ACTIVATION,
#     in_channels = 4,
# )

##model = torch.load('/media/xingshi2/data/matting/Unet3/vgg_current_model.pth')
if os.path.exists(init_pre_path):
    model1 = torch.load(init_pre_path)
    aaa=model1.state_dict()
    model.load_state_dict(aaa, strict=False)

if 1:
    dataset_train = MyDataset(
        x_train_dir,
        y_train_dir,
        y_train_pre_dir,
        train_flag= True,
        classes=CLASSES,
        portraitNet=False,  #edge
        mark4channal = use4channal)   

    valid_dataset = MyDataset( 
        x_valid_dir,
        y_valid_dir,
        y_valid_pre_dir,
        train_flag= False,
        classes=CLASSES,
        portraitNet=False,  #dege
        mark4channal = use4channal)
else:
    dataset_train = MyDataset_select(
        x_train_dir,
        y_train_dir,
        y_train_pre_dir,
        train_flag= True,
        classes=CLASSES,
        portraitNet=False,  #edge
        mark4channal = use4channal)   

    valid_dataset = MyDataset_select( 
        x_valid_dir,
        y_valid_dir,
        y_valid_pre_dir,
        train_flag= False,
        classes=CLASSES,
        portraitNet=False,  #dege
        mark4channal = use4channal)

if winNolinux:
    set_batchsize = 4#args.batchsize
    set_workers = 0
    set_valid_batchsize = 1
    set_valid_workers =0
else:
    set_batchsize = 16  #8
    set_workers = 6
    set_valid_batchsize = 1
    set_valid_workers =1


train_loader = DataLoader(dataset_train, batch_size=set_batchsize, shuffle=True, num_workers=set_workers)
valid_loader = DataLoader(valid_dataset, batch_size=set_valid_batchsize, shuffle=False, num_workers=set_valid_workers)


loss = smp.utils.losses.BCEDiceLoss(eps=1.)
# metrics = [
#     smp.utils.metrics.IoUMetric(eps=1.),
#     smp.utils.metrics.FscoreMetric(eps=1.),
# ]

metrics = [
    smp.utils.metrics.IoU(eps=1.),
    smp.utils.metrics.Fscore(eps=1.),
]

# loss = smp.utils.losses.DiceLoss()
# metrics = [
#     smp.utils.metrics.IoU(threshold=0.5),
# ]

# optimizer = torch.optim.Adam([ 
#     dict(params=model.parameters(), lr=0.0001),
# ])




# optimizer = torch.optim.Adam([
#     {'params': model.decoder.parameters(), 'lr': 1e-4}, 
    
#     # decrease lr for encoder in order not to permute 
#     # pre-trained weights with large gradients on training start
#     {'params': model.encoder.parameters(), 'lr': 1e-4},  
#     {'params': model.segmentation_head.parameters(), 'lr': 1e-3},
# ])


optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': 1e-4}, 
    
    # decrease lr for encoder in order not to permute 
    # pre-trained weights with large gradients on training start
    {'params': model.encoder.parameters(), 'lr': 0},  
    {'params': model.segmentation_head.parameters(), 'lr':  1e-5},
])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=0.0000001)


train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

 
max_score = 0

for i in range(0, 400):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    

    if False:
        valid_logs = valid_epoch.run(valid_loader)
        scheduler.step(valid_logs['iou_score'])
    #    lr = scheduler.get_lr()
        # do something (save model, change lr, etc.)
       
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            model_save_name = save_path+'_%s_%s.pth'%(i,max_score)
            torch.save(model, model_save_name)
            print('Model saved!')
    else: 
        # model_save_name = save_path+'_%s_%s.pth'%(i,max_score)
        model_save_name = save_path+'_%s.pth'%i
        if max_score < train_logs['iou_score']:
            max_score = train_logs['iou_score']
            model_save_name = save_path+'_%s_%s.pth'%(i,max_score)
            torch.save(model, model_save_name)
            print('Model saved!')

    torch.save(model,save_last_path)



#weight of size [128, 269, 3, 3], expected input[16, 288, 60, 80] to have 269 channels, but got 288 channels instead

import time
t1 = time.time()
for i in range(500):
    pr_mask = model.predict(rgbm)

t2 = time.time()

print (500/(t2-t1))

