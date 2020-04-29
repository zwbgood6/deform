import cv2
import numpy as np
import glob
 
img_array = []
for filename in glob.glob('/home/zwenbo/Documents/research/deform/rope_dataset/rope_seg/run05/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('ground_truth.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()