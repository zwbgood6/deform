import numpy as np

add1 = './rope_dataset/clean_dataset/rope_no_loop_seg/'
add3 = '/simplified_clean_actions.npy'

actions_all = None
for i in range(3, 68):
    if len(str(i)) == 1:
        add2 = 'run0{}'.format(i)
    elif len(str(i)) == 2:
        add2 = 'run{}'.format(i)    
    actions = np.load(add1+add2+add3)
    if i == 3:
        actions_all = actions
    else:    
        actions_all = np.append(actions_all, actions, axis=0)

np.save('/home/zwenbo/Documents/research/deform/rope_dataset/clean_dataset/simplified_clean_actions_all_size240.npy', actions_all)