import os
import numpy as np
import sys
sys.path.append("D:/pengt/segmetation/4channels/Unet4/segmentation_models.pytorch-master")
# sys.path.append("/mnt/pentao/code/4channels/Unet4/segmentation_models.pytorch-master")
import cv2
import torch
import torch.backends.cudnn as cudnn
# import segmentation_models_pytorch as smp

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = " "
#linux
# image_path = '/mnt/pentao/data/mydata/val'
# val_image_list = '/mnt/pentao/data/mydata/val/valmask.txt'
# model_path = '/mnt/pentao/code/pytorch-nested-unet/4chmodels/mydata/val_NestedUNet_woDS/model_ori.pth'
# tempyml= '/mnt/pentao/code/pytorch-nested-unet/4chmodels/mydata/val_NestedUNet_woDS/config.yml'
# 
# model_ENCODER =  'resnet101'  #'resnet50'  'resnet152'   101


# model_path = '//SmartGo-Nas/pentao/code/4channels/Unet4/parameter/%s/best_model_last_0.pth'%model_ENCODER
# model_path = 'D:/pengt/code/Cplus/segmentation_human/segmentation_human/model/best_model_21_b.pth'
model_path='D:/pengt/segmetation/4channels/Unet4/parameter_3c/best_model.pth'
# save_p7_path='D:/pengt/code/Cplus/segmentation_human/segmentation_human/model/best_model_21.t7'
# save_net_path='D:/pengt/code/Cplus/segmentation_human/segmentation_human/model/best_model_21.net'

save_script_path = 'D:/pengt/code/Cplus/segmentation_human/segmentation_human/model/Unet3c_320x320_script_eval_cpu.pt'

# ENCODER_WEIGHTS = 'imagenet'
# CLASSES = ['person']
# ACTIVATION = 'sigmoid'
# model = smp.Unet(
#             encoder_name=model_ENCODER,
#             encoder_weights=ENCODER_WEIGHTS,
#             classes=len(CLASSES),
#             activation=ACTIVATION,
#         )
# model = model.cuda()
model = torch.load(model_path,map_location=torch.device('cpu') )

# model.load_state_dict(model_weight)

# model.cuda()
# model.cpu()
model.eval()


# torch.save(model,save_p7_path)
# torch.save(model,save_net_path)
# #######################################
# for param in model.parameters():
# 	param.requires_grad = False
###############################
# 向模型中输入数据以得到模型参数 
example = torch.rand(1,3,320,320)
traced_script_module = torch.jit.trace(model,example)
 
# 保存模型
traced_script_module.save(save_script_path)
# torch.jit.save(model, PATH) #保存
# model = torch.jit.load(model) #读取