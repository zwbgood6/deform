# crop image based on certain designed size

import numpy as np
import cv2
import subprocess

num_range = np.linspace(5, 68, 64, dtype=int)
add1 = '/home/zwenbo/Documents/research/deform/rope_dataset'

for j in num_range:
    if len(str(j)) == 1:
        add2 = 'run0{}'.format(j)
    elif len(str(j)) == 2:
        add2 = 'run{}'.format(j)


    # count number of images in each folder
    add = 'cd ' + add1 + '/rope/' + add2 + '; ls -1 | wc -l'
    num_file = int(subprocess.check_output(add, shell=True)) - 1

    for i in range(num_file):
        if len(str(i)) == 1:
            add3 = '/img_000{}.jpg'.format(i)
        elif len(str(i)) == 2:
            add3 = '/img_00{}.jpg'.format(i)
        elif len(str(i)) == 3:
            add3 = '/img_0{}.jpg'.format(i)  
        elif len(str(i)) == 4:
            add3 = '/img_{}.jpg'.format(i)                        
        image = cv2.imread(add1 + '/rope/' + add2 + add3)
        y=21
        x=0
        h=169
        w=196
        crop = image[y:y+h, x:x+w]
        cv2.imwrite(add1 + '/rope_crop/' + add2 + add3, crop)
