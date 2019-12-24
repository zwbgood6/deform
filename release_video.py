import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# release video
image_folder = '/home/zwenbo/Documents/research/deform/deform/img'
video_name = '/home/zwenbo/Documents/research/deform/deform/video/video_030_v1.avi'

images_temp = [img for img in os.listdir(image_folder)] #if img.endswith(str(i) + ".png")]
images = []
for i in range(len(images_temp)):
    for j in images_temp:
        directory = str(i) + '.png' 
        if directory == j:
            images.append(j)
frame = cv2.imread(os.path.join(image_folder, images_temp[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
   video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()