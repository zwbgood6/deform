# Thresholding the image based on color range

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
    img = cv2.imread(add1 + '/rope_mask/' + add2 + add3)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define ranges for colors in HSV color space you wish to display
    ## LIGHT AND DARK GREEN
    lower_green_light = np.array([20,20,50])
    upper_green_light = np.array([130,150,255])

    ## GREEN
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([70, 255, 255])

    # Threshold with inRange() get only specific colors
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_green_light = cv2.inRange(hsv, lower_green_light, upper_green_light)


    ## union two masks
    mask_union = cv2.bitwise_or(mask_green, mask_green_light)

    ## inverse mask
    mask_union = 255 - mask_union

    # create white image
    white_img = np.zeros(np.shape(img))
    white_img.fill(255)
    ## Perform bitwise operation with the masks and original image
    res_green_light = cv2.bitwise_and(white_img,white_img, mask=mask_union)
    cv2.imwrite(add1 + '/rope_seg/' + add2 + add3, res_green_light)

 