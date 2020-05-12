import numpy as np

add1 = './rope_dataset/rope/'
add2 = 'run'
add3 = '/actions.npy'

actions_all = None
for i in range(3, 69):
    if len(str(i)) == 1:
        add2 = 'run0{}'.format(i)
    elif len(str(i)) == 2:
        add2 = 'run{}'.format(i)    
    actions = np.load(add1+add2+add3)
    if i == 3:
        actions_all = actions
    else:    
        actions_all = np.append(actions_all, actions, axis=0)

np.save('/home/zwenbo/Documents/research/deform/rope_dataset/rope_all/actions.npy', actions_all)