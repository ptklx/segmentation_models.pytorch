#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
#import cv2
#import matplotlib.pyplot as plt
import  sys
sys.path.append('/home/mydisk/code/matting/Unet4')
sys.path.append('/home/mydisk/code/matting/Unet4/segmentation_models.pytorch-master')
import segmentation_models_pytorch as smp
import albumentations as albu
import torch
import time
from torch.autograd import Variable

import torch
print(torch.version.__version__)

#384x512

#pth_path = '/home/pt/Desktop/code/matting/traindata/unet4/yxl_1/res18_yxl.pth'
pth_path = '/home/mydisk/code/matting/traindata/unet4/best_model.pth'


model = torch.load(pth_path)
model=model.cuda()

#dummy_input = Variable(torch.randn(1, 3, 224 , 224)).cpu()
#dummy_input = torch.randn(1, 4, 320, 320, requires_grad=True)
model=model.cpu()
print(model)


#model = torchvision.models.resnet18(pretrained=True)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("/home/mydisk/code/androidA/android-demo-app/HelloWorldApp/app/src/main/assets/my_model.pt")
    
    