# delete loop image associated action
import numpy as np
import os 

add1 = '/home/zwenbo/Documents/research/deform/rope_dataset/simplified_dataset/rope_no_loop'

for j in range(62, 69):
    if len(str(j)) == 1:
        add2 = '/run0{}'.format(j)
    elif len(str(j)) == 2:
        add2 = '/run{}'.format(j) 
    actions = np.load(add1 + add2 + '/actions.npy')
    n, _ = np.shape(actions)
    simplified_actions = []

    for i in range(n):
        if len(str(i)) == 1:
            add3 = '/img_000{}.jpg'.format(i)
        elif len(str(i)) == 2:
            add3 = '/img_00{}.jpg'.format(i)
        elif len(str(i)) == 3:
            add3 = '/img_0{}.jpg'.format(i) 
        elif len(str(i)) == 4:
            add3 = '/img_{}.jpg'.format(i) 

        if os.path.exists(add1 + add2 + add3):
            simplified_actions.append(actions[i])    

    np.save(add1 + add2 + '/simplified_actions.npy', simplified_actions)
