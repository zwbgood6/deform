from PIL import Image 
import cv2
import numpy as np

for i in range(2353):
    add1 = '/home/zwenbo/Documents/research/deform/rope_dataset'
    add2 = 'run05' # CHANGE
    if len(str(i)) == 1:
        add3 = '/img_000{}.jpg'.format(i)
    elif len(str(i)) == 2:
        add3 = '/img_00{}.jpg'.format(i)
    elif len(str(i)) == 3:
        add3 = '/img_0{}.jpg'.format(i) 
    elif len(str(i)) == 4:
        add3 = '/img_{}.jpg'.format(i) 

    # Read the image and transform it to HSV color space
    img = cv2.imread(add1 + '/rope_seg/' + add2 + add3)
    img = cv2.resize(img, (50,50), interpolation = cv2.INTER_AREA)
    add2 = 'run05_resize'
    cv2.imwrite(add1 + '/rope_seg/' + add2 + add3, img)

