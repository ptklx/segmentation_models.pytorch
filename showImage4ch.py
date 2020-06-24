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
image_path = 'D:/pengt/data/mydata/val'
val_image_list = 'D:/pengt/data/mydata/val/valmask.txt'
model_ENCODER =  'resnet50'  #'resnet50'

# model_path = '//SmartGo-Nas/pentao/code/4channels/Unet4/parameter/%s/best_model_last_0.pth'%model_ENCODER
model_path = '//SmartGo-Nas/pentao/code/4channels/Unet4/parameter_last/%s/best_model_0.pth'%model_ENCODER

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
 

img_width = 640
img_height = 480

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


    # Data loading code
    if False:
        img_ids = glob(os.path.join(image_path, 'images', '*.jpg'))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    else:
        #read txt
        fi = open(val_image_list, 'r') 
        # img_ids = fi.readlines()
        # img_ids = img_ids[0][:-1]
        img_ids = fi.read().splitlines()
        fi.close()

    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    # avg_meter = AverageMeter()
    # for c in range(config['num_classes']):
    #     os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    pre_mask = np.zeros((img_height,img_width))
    # os.makedirs(os.path.join('outputs', config['name']), exist_ok=True)
    with torch.no_grad():
        #for id_tx in img_ids:
        for index, id_tx in enumerate(img_ids):
            img_id = id_tx
        
            img = cv2.imread(os.path.join(image_path,"images", img_id + ".jpg"))
            mask=cv2.imread(os.path.join(image_path,"masks",img_id + ".png"), cv2.IMREAD_GRAYSCALE)[..., None]
            bgr = cv2.resize(img, ( img_width,img_height))
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
            # with torch.no_grad():
            #    prediction = model.forward(rgbd)
            # pr_mask = pr_mask.squeeze().cpu().numpy().round()
            pr_mask = pr_mask.squeeze().cpu().numpy()
            # print(pr_mask)
            pr_mask = (pr_mask > 0.5).astype(np.uint8)

            # target = cv2.resize(mask, (img_width,img_height))
            # result_out = output[0][0,:,:]
            # iou = iou_score(result_out, target)
            # avg_meter.update(iou, img1.size(0))
            # result_out = torch.sigmoid(result_out).cpu().numpy()
            # # result_out =result_out.cpu().numpy()
            # img_new_seg = (result_out >0.5).astype(np.uint8)
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
            cv2.waitKey(5)
            print("index%s"%id_tx)
            # if 0:
            #     for i in range(len(output)):
            #         for c in range(config['num_classes']):
            #             cv2.imwrite(os.path.join('outputs', config['name'],  meta['img_id'][i] + '.jpg'),
            #                         (output[i, c] * 255).astype('uint8'))

    # print('IoU: %.4f' % avg_meter.avg)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
