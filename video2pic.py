import os
import numpy as np
import sys

import cv2

#linux
# image_path = '/mnt/pentao/data/mydata/val'
# val_image_list = '/mnt/pentao/data/mydata/val/valmask.txt'

#win10
# video_path = '//SmartGo-Nas/pentao/data/video/111'
# out_path='//SmartGo-Nas/pentao/data/video/small_pic'

video_path = '/mnt/pentao/data/video/111'
out_path='/mnt/pentao/data/video/small_pic'
###########
#%%




def main():
    mask = cv2.imread("//SMARTGO-NAS/pentao/data/my_cleardata/baidu_v1/bighard/masks/13.png",0)
    mask[mask>0] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5)) 
    mask0 =  cv2.erode(mask,kernel,iterations=3)
    mask1 =  cv2.erode(mask0,kernel,iterations=5)
    # cv2.imshow("erode1",mask0)
    # cv2.imshow("erode2",mask1)
    mask = mask*200
    mask0 = mask0*200
    mask1 = mask1*200
    last = cv2.merge([mask,mask0, mask1])
    cv2.imshow("last",last)
    cv2.waitKey(0)
    mask = mask[None]   #在前面增加一维
    mask[mask!=0]=1

    ################################

    ve_list = os.listdir(video_path)
    n_b = 0 
    for v_name in ve_list:               
        videofi = os.path.join(video_path,v_name)
        outwholepath = os.path.join(out_path,os.path.splitext(v_name)[0])
        if  not os.path.exists(outwholepath):
            os.makedirs(outwholepath)
        print(n_b,"###")
        # if n_b != 2 :      #select video
        #     n_b+=1
        #     continue
        n_b+=1
        
        cap=cv2.VideoCapture(videofi)
        index =0
        while (True):
            ret,frame=cap.read()  
            index+=1
            startindex = 8000
            second = 16000
            third = 18000
            countNum = 200
            if index <startindex:   
                continue
            if index>startindex+countNum:     # 6000
                if index<second:
                    continue
            if index> second+countNum:
                if index<third:
                    continue
            if index > third+countNum:
                break

                

            filename = os.path.join(outwholepath,"%d_%d.jpg"%(index,n_b))
            
            cv2.imwrite(filename, frame)
            # cv2.imshow("mask",img_new_seg*200)
            # cv2.imshow("bgr",bgr)
            # cv2.waitKey(5)
            print("index_%s"%index)
                    



if __name__ == '__main__':
    main()
