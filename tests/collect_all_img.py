import numpy as np
import cv2
import subprocess

num_range = np.linspace(3, 68, 66, dtype=int)
add1 = '/home/zwenbo/Documents/research/deform/rope_dataset'

count = -1

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
        img = cv2.imread(add1 + '/rope/' + add2 + add3)
        # rename image
        cv2.imwrite(add1 + '/rope_all_resize_ori/' + add4, img)
