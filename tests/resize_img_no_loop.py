from PIL import Image 
import cv2
import numpy as np

for i in range(22515):
    add1 = '/home/zwenbo/Documents/research/deform/rope_dataset/simplified_dataset/rope_no_loop_all_gray'
    if len(str(i)) == 1:
        add3 = '/img_0000{}.jpg'.format(i)
    elif len(str(i)) == 2:
        add3 = '/img_000{}.jpg'.format(i)
    elif len(str(i)) == 3:
        add3 = '/img_00{}.jpg'.format(i) 
    elif len(str(i)) == 4:
        add3 = '/img_0{}.jpg'.format(i) 
    elif len(str(i)) == 5:
        add3 = '/img_{}.jpg'.format(i) 

    # Read the image 
    img = cv2.imread(add1 + add3)
    # resize the image
    img = cv2.resize(img, (50,50), interpolation = cv2.INTER_AREA)
    # image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    add2 = '/home/zwenbo/Documents/research/deform/rope_dataset/simplified_dataset/rope_no_loop_all_resize_gray'
    #cv2.imwrite(add2 + add3, img)

