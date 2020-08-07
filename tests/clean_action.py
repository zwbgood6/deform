import numpy as np
import subprocess
import os

add1 = '/home/zwenbo/Documents/research/deform/rope_dataset/clean_dataset/rope_no_loop_seg/'
add = '/home/zwenbo/Documents/research/deform/rope_dataset/rope/'

for j in range(3, 68):
    if len(str(j)) == 1:
        add2 = 'run0{}'.format(j)
    elif len(str(j)) == 2:
        add2 = 'run{}'.format(j)

    add3 = 'cd ' + add + add2 + '; ls -1 | wc -l'
    num_file = int(subprocess.check_output(add3, shell=True)) - 1  
    action_path = add1 + add2 + '/simplified_actions.npy'
    actions = np.load(action_path)     
    index = 0

    for i in range(num_file):        
        if len(str(i)) == 1:
            add4 = '/img_000{}.jpg'.format(i)
        elif len(str(i)) == 2:
            add4 = '/img_00{}.jpg'.format(i)
        elif len(str(i)) == 3:
            add4 = '/img_0{}.jpg'.format(i) 
        elif len(str(i)) == 4:
            add4 = '/img_{}.jpg'.format(i)            

        if len(str(i+1)) == 1:
            add5 = '/img_000{}.jpg'.format(i+1)
        elif len(str(i+1)) == 2:
            add5 = '/img_00{}.jpg'.format(i+1)
        elif len(str(i+1)) == 3:
            add5 = '/img_0{}.jpg'.format(i+1) 
        elif len(str(i+1)) == 4:
            add5 = '/img_{}.jpg'.format(i+1)  

        if not os.path.exists(add1 + add2 + add4):
            continue   
        index += 1
        if os.path.exists(add1 + add2 + add4) and (not os.path.exists(add1 + add2 + add5)):
            actions[index-1][4] = 0
    
    new_action_path = add1 + add2 + '/simplified_clean_actions.npy'
    np.save(new_action_path, actions)        