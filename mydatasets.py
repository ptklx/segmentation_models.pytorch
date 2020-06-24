import sys
import os
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import functools
# import keras
from torch.utils.data import Dataset as BaseDataset

#10    11   12  13   14    15  16    20   24   25   30   
#   *32
#320  352  384 416  448   480   512  640  786  800   960

padheight = 512
padwidth = 640

inputheight = 480
inputwidth = 640



def preprocess_input(x, mean=None, std=None, input_space='RGB', input_range=None, **kwargs):
    if input_space == 'BGR':
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std
    return x


#4channal
# import segmentation_models_pytorch as smp
# preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
preprocessing_fn_4ch = functools.partial(preprocess_input, mean=[0.485, 0.456, 0.406, 0], std=[0.229, 0.224, 0.225, 1], input_space='RGB', input_range=[0,1])

preprocessing_fn = functools.partial(preprocess_input, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], input_space='RGB', input_range=[0,1])

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        if len(image.shape)==3 and image.shape[2]==4:
            n = n+1
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image[:,:,0])
            plt.subplot(1, n, i + 3)
            plt.xticks([])
            plt.yticks([])
            str1 = name.split('_')
            str1.append('4c')
            plt.title(' '.join(str1).title())
            plt.imshow(image[:,:,3])
        else:
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)
        
    plt.show()
    # cv2.waitKey(0)



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

def get_preprocessing_4ch(preprocessing_fn_4ch):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn_4ch),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)



#  albu.Resize(height= padheight, width= padwidth),  如果降低人体部分
def get_training_augmentation():
    train_transform = [
        
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.2),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=50, shift_limit=0.1, p=1, border_mode=0),
        #320 384  448 512  640
#        albu.GridDistortion(num_steps=2, distort_limit=0.2, interpolation=1, border_mode=0, value=None, always_apply=False, p=0.5),
        albu.PadIfNeeded(min_height=padheight, min_width=padwidth, always_apply=True, border_mode=0),
        albu.Resize(height= padheight, width= padwidth),
        albu.RandomCrop(height=inputheight, width=inputwidth, always_apply=True),   #the last size
        
        albu.ChannelShuffle(p=0.1),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

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



def get_next_augmentation():
    train_transform = [
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
        albu.PadIfNeeded(min_height=padheight, min_width=padwidth,border_mode = cv2.BORDER_CONSTANT),
        albu.RandomCrop(height=inputheight, width=inputwidth, always_apply=True),
    ]
    return albu.Compose(test_transform)



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

def show_edge(mask_ori):
    if len(mask_ori.shape)==3:
        if mask_ori.shape[2]==1:
            mask = mask_ori[:,:,0].copy()
        else:
            mask = mask_ori[0].copy()
    else:
        mask = mask_ori.copy()
    # find countours: img must be binary
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
    cv2.drawContours(myImg, countours, -1, 1, 4)
    return myImg



def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

class MyDataset(BaseDataset):
# class MyDataset(keras.utils.Sequence):
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
            pre_mask_dir = None,
            classes=None, 
            train_flag = False,           
            preprocessing= None,#get_preprocessing(preprocessing_fn),
            portraitNet=False,
            mark4channal = True,

    ):  
        self.mark4channal = mark4channal
        self.ids = os.listdir(images_dir) 
        self.images_fps =[]
        self.masks_fps =[]
        for  image_id in self.ids:
            if os.path.splitext(image_id)[1] =='.jpg':   #######
                self.images_fps.append(os.path.join(images_dir, image_id))
                self.masks_fps.append(os.path.join(masks_dir,image_id))

        # self.images_fps = [os.path.join(images_dir, image_id) for  image_id in self.ids]
        # self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        

        #self.img_len = len(self.ids)
        self.img_len = len(self.images_fps)
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        if train_flag:
            self.augmentation = get_training_augmentation()
        else:
            self.augmentation = get_validation_augmentation()
        self.next_augmentation = get_next_augmentation()
       
        if preprocessing!=None:
            self.preprocessing = preprocessing
        else:
            if mark4channal:
                self.preprocessing =get_preprocessing_4ch(preprocessing_fn_4ch)
            else:
                self.preprocessing =get_preprocessing(preprocessing_fn)

        self.portraitNet = portraitNet
        
        if pre_mask_dir is not None:
            self.pre_mask_dir = pre_mask_dir
            self.pre_masks_fps=[]
            for image_id in self.ids:
                if os.path.splitext(image_id)[1]=='.jpg':
                    self.pre_masks_fps.append(os.path.join(pre_mask_dir, image_id))
             #self.pre_masks_fps = [os.path.join(pre_mask_dir, image_id) for image_id in self.ids]
        else:
            self.pre_mask_dir = None
            self.pre_masks_fps = None
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        #if image is None:
        #print(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # self.masks_fps[i].replace('.jpg', '.png')
        mask = cv2.imread(self.masks_fps[i][:-4]+'.png', 0)
        #print(self.masks_fps[i][:-4]+'.png')
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        mask[mask == 0] = 2
        mask = mask - 1
        # apply augmentations
        if self.augmentation:
            if self.pre_mask_dir is not None:
                mask0 = cv2.imread(self.pre_masks_fps[i][:-4]+'.png', 0)
                mask0[mask0 > 0] = 128
                mask0 = mask0[:,:,np.newaxis]
                sample = get_validation_augmentation()(image=image, mask=mask0)
                #sample = self.augmentation(image=image, mask=mask,mask0=mask0)
                image, mask0 = sample['image'], sample['mask']
                mask = get_validation_augmentation()(image=mask)['image']
                image = np.concatenate((image,mask0),-1)
            else:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
                if self.portraitNet:
                    next_image=self.next_augmentation(image = image)['image']
        
        if self.pre_mask_dir is None:       
            p = 0.7
            rand_num = random.uniform(0,1.)
            mask0 = mask.copy()
            
            h,w,_ = np.shape(mask0)
            if rand_num < p:
#                print ('qqqqqqqqqqqqq:',rand_num)
                r_scale = random.uniform(0.9, 1.1)
                r_shift_w,r_shift_h = int(w*random.uniform(-0.08,0.08)),int(h*random.uniform(-0.08,0.08))
                r_rotation = random.randint(-30,30)
                
                A2 = np.array([[r_scale, 0, r_shift_w], [0, r_scale, r_shift_h]], np.float32)
                mask0 = cv2.warpAffine(mask0, A2, (w, h), borderValue=0)
                
                A3 = cv2.getRotationMatrix2D((w/2.0, h/2.0), r_rotation, 1)
                mask0 = cv2.warpAffine(mask0, A3, (w, h), borderValue=0)  
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5)) 
                
                
                if random.uniform(0,1.) < 0.5:  
#                    print ('a1')
#                    print (np.shape(mask0))
                    mask0 = cv2.erode(mask0,kernel,iterations=5)
                    mask0 = cv2.dilate(mask0,kernel,iterations=4)
                
                if random.uniform(0,1.) < 0.5:   #读取另外的帧增加干扰
#                    print ('a2')
#                    print (np.shape(mask0))
                    rrr = random.randint(0,self.img_len-1)
                    mask1 = cv2.imread(self.masks_fps[rrr][:-4]+'.png', 0)
                    #print("add")
                    #print(self.masks_fps[rrr][:-4]+'.png')
                    mask1 = cv2.erode(mask1,kernel,iterations=3)  ####
                    mask1 = cv2.resize(mask1,(w,h))
                    mask0 += mask1  
                #########add20200509
                if random.uniform(0,1.) < 0.3:      
                    mask0 = cv2.erode(mask0,kernel,iterations=3)
                if random.uniform(0,1.) < 0.3:
                    mask0 = cv2.dilate(mask0,kernel,iterations=3)

            else:
              
                mask0 = mask*0
                mask0 = mask0[:,:,0]
#                print ('a3')
#                print (np.shape(mask0))
       
            mask0[mask0 > 0] = 128     #####################################value
            mask0 = mask0[:,:,np.newaxis]
#            print ('aaaaa')
#            print (np.shape(mask0))
            # plt.imshow(mask0[:,:,-1])
            if self.mark4channal:
                #np.c_[image, mask0]
                image = np.concatenate((image,mask0),-1)
                if self.portraitNet:
                    next_image = np.concatenate((next_image,mask0),-1)
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            if self.portraitNet:
                sample = self.preprocessing(image=next_image)
                next_image = sample['image']

        mask[mask!=0] = 1 

        if self.portraitNet:
            edge = show_edge(mask)
            if False:
                cv2.imshow("test",edge.astype(np.uint8)*255)
                cv2.waitKey(0)
            edge = np.stack(edge, axis=-1).astype(mask.dtype)
            edge = edge[None]   #在前面增加一维
            edge[edge!=0]=1
            return next_image,image,edge,mask
        else:
            return image, mask
        
    def __len__(self):
        #return len(self.ids)
        return len(self.images_fps)


if __name__ == "__main__":
    
    sys.path.append("./data")

    DATA_DIR = '//SmartGo-Nas/pentao/data/mydata'
    x_train_dir = os.path.join(DATA_DIR, 'zdyy_clear/images')
    y_train_dir = os.path.join(DATA_DIR, 'zdyy_clear/masks')

    x_valid_dir = os.path.join(DATA_DIR, 'zdyy_clear/images')
    y_valid_dir = os.path.join(DATA_DIR, 'zdyy_clear/masks')

    x_test_dir = os.path.join(DATA_DIR, 'zdyy_clear/images')
    y_test_dir = os.path.join(DATA_DIR, 'zdyy_clear/masks')

    portraitNet_flag = True
    augmented_dataset = MyDataset(
    x_train_dir, 
    y_train_dir, 
    train_flag= True,
    classes=['person'],
    portraitNet=portraitNet_flag,
    mark4channal = False,

    )
    #####
    #如果调试squeeze(-1) 需要设置 None        
    # preprocessing=None,#get_preprocessing(preprocessing_fn),
    if False:
        if  portraitNet_flag == False:
            for i in range(5):
                image, mask = augmented_dataset[100+i]
                visualize(image=image, mask=mask.squeeze(-1))   
        else:
            for i in range(5):
                next_image,image,edge, mask = augmented_dataset[100+i]
                visualize(next_image=next_image,image=image, mask=mask.squeeze(-1))  
    else:
        if  portraitNet_flag == False:
            for i in range(5):
                image, mask = augmented_dataset[100+i]
                visualize(image=np.transpose(image, (1, 2, 0)), mask=mask.squeeze(0))   
        else:
            for i in range(5):
                next_image,image,edge, mask = augmented_dataset[100+i]
                visualize(next_image=np.transpose(next_image, (1, 2, 0)) ,image=np.transpose(image, (1, 2, 0)), mask=mask.squeeze(0)) 




    # ####
    # dataset = MyDataset(x_train_dir, y_train_dir, classes=['person'])

    
    # k=221
    # k+=1
    # image, mask = dataset[k] # get some sample
    # visualize(
    #    image=image[:,:,3].astype('uint8'), 
    #    cars_mask=mask.squeeze(),
    # )
