# copy action from one folder to another folder
import numpy as np
import shutil

add1 = '/home/zwenbo/Documents/research/deform/rope_dataset/simplified_dataset/rope_no_loop/'
add3 = '/home/zwenbo/Documents/research/deform/rope_dataset/simplified_dataset/rope_no_loop_seg/'

for j in range(3, 68):
    if len(str(j)) == 1:
        add2 = 'run0{}'.format(j)
    elif len(str(j)) == 2:
        add2 = 'run{}'.format(j)
    
    newpath = shutil.copy(add1+add2+'/simplified_actions.npy', add3+add2)