import numpy as np
import cv2
import subprocess
import os

#num_range = np.linspace(3, 68, 66, dtype=int)
add1 = '/home/zwenbo/Documents/research/deform/rope_dataset'

count = 0

for j in range(3, 68):
    if len(str(j)) == 1:
        add2 = '/run0{}'.format(j)
    elif len(str(j)) == 2:
        add2 = '/run{}'.format(j)

    # count number of images in each folder
    # add = 'cd ' + add1 + add2 + '; ls -1 | wc -l'
    # num_file = int(subprocess.check_output(add, shell=True)) 

    for i in range(9999):
        if len(str(i)) == 1:
            add3 = '/img_000{}.jpg'.format(i)
        elif len(str(i)) == 2:
            add3 = '/img_00{}.jpg'.format(i)
        elif len(str(i)) == 3:
            add3 = '/img_0{}.jpg'.format(i) 
        elif len(str(i)) == 4:
            add3 = '/img_{}.jpg'.format(i)         
        
        if not os.path.exists(add1 + '/rope_clean' + add2 + add3):
            continue

        
        if len(str(count)) == 1:
            add4 = '/img_0000{}.jpg'.format(count)
        elif len(str(count)) == 2:
            add4 = '/img_000{}.jpg'.format(count)
        elif len(str(count)) == 3:
            add4 = '/img_00{}.jpg'.format(count) 
        elif len(str(count)) == 4:
            add4 = '/img_0{}.jpg'.format(count)  
        elif len(str(count)) == 5:
            add4 = '/img_{}.jpg'.format(count)   

        count += 1
        # read image
        img = cv2.imread(add1 + '/rope_clean' + add2 + add3)
        # rename image
        cv2.imwrite(add1 + '/rope_clean_all' + add4, img)
