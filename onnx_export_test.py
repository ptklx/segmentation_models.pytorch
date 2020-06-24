#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
# import cv2
# import matplotlib.pyplot as plt
import  sys
# import segmentation_models_pytorch as smp
import albumentations as albu
import torch
import time
from torch.autograd import Variable
import onnx

#384x512

#pth_path = '/home/pt/Desktop/code/matting/traindata/unet4/yxl_1/res18_yxl.pth'
# pth_path = 'D:/pengt/segmetation/4channels/Unet4/parameter/best_model.pth'

# onnx_out='D:/pengt/segmetation/4channels/Unet4/parameter/onnx_best_model.onnx'

model_ENCODER = 'timm-efficientnet-b0'# 'timm-efficientnet-b0' #'dpn68'#'resnet18' #'mobilenet_v2' #'resnet18'# 'resnet101'  #'resnet50'  'resnet152'   101


model_path = '//SmartGo-Nas/pentao/code/4channels/parameter/seg_pytorch/%s/best_model_52_0.936.pth'%model_ENCODER

# onnx_out='D:/pengt/code/Cplus/onnx_model/deeplab/%s_384x640_new1.onnx'%model_ENCODER

onnx_out='D:/pengt/code/Cplus/onnx_model/deeplab1/%s_640x384.onnx'%model_ENCODER

# ENCODER= 'densenet121'#'vgg13' #timm-efficientnet-b0'  #"mobilenet_v2"  

# ENCODER_WEIGHTS = 'imagenet'
# CLASSES = ['person']
# ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation

# model = smp.FPN(
#     encoder_name=ENCODER, 
#     encoder_weights=ENCODER_WEIGHTS, 
#     classes=len(CLASSES), 
#     activation=ACTIVATION,
# )

# model = smp.DeepLabV3Plus(
#     encoder_name=model_ENCODER, 
#     encoder_weights=ENCODER_WEIGHTS, 
#     classes=len(CLASSES), 
#     activation=ACTIVATION,
#     in_channels = 4,
# )


# model = torch.load(model_path,map_location=torch.device('cuda'))
model = torch.load(model_path)
model=model.cuda()




dummy_input = Variable(torch.randn(1, 4, 640 , 384))
# #dummy_input = torch.randn(1, 4, 320, 320, requires_grad=True)
model=model.cpu()


# dummy_input = Variable(torch.randn(1, 4, 480 , 640))
#dummy_input = torch.randn(1, 4, 320, 320, requires_grad=True)



model.eval()
print(model)
#torch.onnx.export(model, dummy_input, "/media/xingshi2/data/purning/network-slimming/Unet_resnet50.onnx", verbose=True)

#model1 = onnx.load("/media/xingshi2/data/purning/network-slimming/Unet_resnet50.onnx")

# Check that the IR is well formed
#onnx.checker.check_model(model1)

# Print a human readable representation of the graph

#onnx.helper.printable_graph(model1.graph)
#opset_version = model1.opset_import[0].version

# model.set_swish(memory_efficient=False)

torch.onnx.export(model,               # model being run
                  dummy_input,                         # model input (or a tuple for multiple inputs)
                  onnx_out,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,#12,          # the onnx version to export the model to
                  verbose=True,
#                  do_constant_folding=True,  # wether to execute constant folding for optimization
                   #input_names = ['final_conv'],   # the model's input names
                   #output_names = ['final_conv'], # the model's output names
#                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
#                                'output' : {0 : 'batch_size'}}
                    
                )    


    
    
    
    