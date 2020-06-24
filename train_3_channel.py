#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append("D:/pengt/segmetation/4channels/Unet4/segmentation_models.pytorch-master")
import numpy as np
import cv2
import matplotlib.pyplot as plt
print(os.getcwd())



# linux 
#DATA_DIR = '/mnt/pentao/data/mydata'

#win10
DATA_DIR = 'D:/pengt/data/mydata'

if True:
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,4'
    # savepath = 'D:/pengt/segmetation/4channels/Unet4/parameter_3c/best_model.pth'
    # DATA_DIR = '/home/mydisk/datapic/mydata_v2'

    x_train_dir = os.path.join(DATA_DIR, 'train/images')
    y_train_dir = os.path.join(DATA_DIR, 'train/masks')

    x_valid_dir = os.path.join(DATA_DIR, 'val/images')
    y_valid_dir = os.path.join(DATA_DIR, 'val/masks')

else:
   
    savepath = '/home/mydisk/code/matting/traindata/unet4/best_model.h5'
    # DATA_DIR = '/home/diskpic/picdata/koutu_data/train'

    x_train_dir = os.path.join(DATA_DIR, 'cheshi/images')
    y_train_dir = os.path.join(DATA_DIR, 'cheshi/masks')

    x_valid_dir = os.path.join(DATA_DIR, 'cheshi/images')
    y_valid_dir = os.path.join(DATA_DIR, 'cheshi/masks')



#y_valid_dir

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset



class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['person']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i][:-4]+'.png', 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        mask[mask == 0] = 2
        mask = mask - 1
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        mask[mask!=0] = 1    
        return image, mask
        
    def __len__(self):
        return len(self.ids)


#dataset = Dataset(x_train_dir, y_train_dir, classes=['person'])
#
#image, mask = dataset[267] # get some sample
#visualize(
#    image=image, 
#    cars_mask=mask.squeeze(),
#)
#

import albumentations as albu


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.2),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=50, shift_limit=0.1, p=1, border_mode=0),

#        albu.GridDistortion(num_steps=2, distort_limit=0.2, interpolation=1, border_mode=0, value=None, always_apply=False, p=0.5),
        albu.PadIfNeeded(min_height=480, min_width=480, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320 ,always_apply=True),
        

        
        albu.ChannelShuffle(p=0.1),
        
        
        albu.IAAAdditiveGaussianNoise(p=0.2),
      

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(320,320,border_mode = cv2.BORDER_CONSTANT),
        albu.RandomCrop(height=320, width=320 , always_apply=True),
    ]
    return albu.Compose(test_transform)


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

augmented_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    classes=['person'],
)

# same image with different random transforms
#for i in range(3):
#    image, mask = augmented_dataset[7618]
#    visualize(image=image, mask=mask.squeeze(-1))


import torch
import numpy as np
import  sys
# sys.path.append('/home/mydisk/code/matting/Unet4')
# sys.path.append('/home/mydisk/code/matting/Unet4/segmentation_models.pytorch-master')
from segmentation_models_pytorch.unet import model 
import  segmentation_models_pytorch as smp


#ENCODER = 'se_resnet50'
ENCODER = 'resnet18'#'inceptionresnetv2' #'resnet34'    #vgg11   vgg11_bn  inceptionresnetv2  'resnet18' resnet34  ''
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

CLASSES = ['person']
ACTIVATION = 'sigmoid'
flagt = 'last_'
# D:/pengt/segmetation/4channels/Unet4/parameter_3c/best_model.pth
save_path = 'D:/pengt/segmetation/4channels/Unet4/parameter_3c/%s_%s/best_model'%(ENCODER,flagt)
save_last_path = 'D:/pengt/segmetation/4channels/Unet4/parameter_3c/%s_%s/best_model_last.pth'%(ENCODER,flagt)

folder_path, file_name = os.path.split(save_path)

# if is not ospath.exists(folder_path)
os.makedirs(folder_path, exist_ok=True)


model = model.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)
##model = torch.load('/media/xingshi2/data/matting/Unet3/vgg_current_model.pth')
#model1 = torch.load('/media/xingshi2/data/matting/Unet3/vgg_yxl_currnet_model.pth')
#aaa=model1.state_dict()
#model.load_state_dict(aaa, strict=False)
import torch.nn as nn


preprocessing_fn = smp.encoders.get_preprocessing_fn('vgg19_bn', 'imagenet')

train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=8 ,shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)



# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = smp.utils.losses.BCEDiceLoss(eps=1.)
metrics = [
    smp.utils.metrics.IoUMetric(eps=1.),
    smp.utils.metrics.FscoreMetric(eps=1.),
]

optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': 1e-4}, 
    
    # decrease lr for encoder in order not to permute 
    # pre-trained weights with large gradients on training start
    {'params': model.encoder.parameters(), 'lr': 1e-4},  
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
        scheduler.step(valid_logs['iou'])
    #    lr = scheduler.get_lr()
        # do something (save model, change lr, etc.)
        model_save_name = save_path+'_%s.pth'%i
        if max_score < valid_logs['iou']:
            max_score = valid_logs['iou']
            torch.save(model, model_save_name)
            print('Model saved!')
    else: 
        scheduler.step(train_logs['iou'])
        model_save_name = save_path+'_%s.pth'%i
        if max_score < train_logs['iou']:
            max_score = train_logs['iou']
            torch.save(model, model_save_name)
            print('Model saved!')

    torch.save(model,save_last_path)




#weight of size [128, 269, 3, 3], expected input[16, 288, 60, 80] to have 269 channels, but got 288 channels instead

import time
t1 = time.time()
for i in range(500):
    pr_mask = model.predict(rbg)

t2 = time.time()

print (500/(t2-t1))

