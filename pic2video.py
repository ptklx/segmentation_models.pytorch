import os
import cv2

outpath = "D:/pengt/data/webvideo/zhoujielu/joiner_mask_bz_ta.avi"  #out
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
# fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
out = cv2.VideoWriter(outpath,fourcc, 23.0, (1280,720),True)
inpic_path = "D:/pengt/data/webvideo/zhoujielu/outpic3"
img_folds_list = os.listdir(inpic_path)

# while (True):
for sub0 in img_folds_list:
    img_path = os.path.join(inpic_path,sub0)
    frame = cv2.imread(img_path)
    # im = cv2.imread("image.png", cv2.IMREAD_UNCHANGED)
    # out.write(result1.astype(np.uint8))
    out.write(frame)

out.release()  ####          