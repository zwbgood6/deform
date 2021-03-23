import numpy as np
import cv2
import subprocess
import os

add1 = '/home/zwenbo/Documents/research/deform/rope_dataset/rope_clean'

count = 0

for j in range(64, 68):
    if len(str(j)) == 1:
        add2 = '/run0{}'.format(j)
    elif len(str(j)) == 2:
        add2 = '/run{}'.format(j)

    # count number of images in each folder
    add = 'cd ' + add1 + add2 + '; ls -1 | wc -l'
    num_file = int(subprocess.check_output(add, shell=True)) 
    count += num_file

print(count)    