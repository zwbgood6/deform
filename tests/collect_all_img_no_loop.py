import numpy as np
import cv2
import subprocess
import os

#num_range = np.linspace(3, 68, 66, dtype=int)
add1 = '/home/zwenbo/Documents/research/deform/rope_dataset/simplified_dataset'

count = -1

# for j in num_range:
for j in range(3, 69):
    if len(str(j)) == 1:
        add2 = 'run0{}'.format(j)
    elif len(str(j)) == 2:
        add2 = 'run{}'.format(j)

    # count number of images in each folder
    add = 'cd ' + '/home/zwenbo/Documents/research/deform/rope_dataset/rope/' + add2 + '; ls -1 | wc -l'
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

        if not os.path.exists(add1 + '/rope_no_loop_seg/' + add2 + add3):
            continue

        count += 1
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

        # read image
        img = cv2.imread(add1 + '/rope_no_loop_seg/' + add2 + add3)
        # rename image
        cv2.imwrite(add1 + '/rope_no_loop_all_gray/' + add4, img)
