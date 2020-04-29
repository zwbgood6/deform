import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# release video
image_folder = '/home/zwenbo/Documents/research/deform/result/test_three_loss/prediction_step2'
video_name = '/home/zwenbo/Documents/research/deform/two_step_prediction.avi'

images_temp = [img for img in os.listdir(image_folder)] #if img.endswith(str(i) + ".png")]
images = []
for i in range(len(images_temp)):
    for j in images_temp:
        directory = 'predict_' + str(i) + '.png'
        # if len(str(i)) == 1:  
        #     directory = 'img_000' + str(i) + '.jpg'
        # if len(str(i)) == 2:
        #     directory = 'img_00' + str(i) + '.jpg'
        # if len(str(i)) == 3:
        #     directory = 'img_0' + str(i) + '.jpg'
        # if len(str(i)) == 4:        
        #     directory = 'img_' + str(i) + '.jpg' 
        if directory == j:
            images.append(j)
frame = cv2.imread(os.path.join(image_folder, images_temp[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
   video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()