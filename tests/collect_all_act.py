import numpy as np

add1 = './rope_dataset/rope/'
add2 = 'run'
add3 = '/actions.npy'

#np.linspace(3, 68, 66, dtype=int)
act_size = []
act_all = 0
for i in range(3, 69):
    if len(str(i)) == 1:
        add2 = 'run0{}'.format(i)
    elif len(str(i)) == 2:
        add2 = 'run{}'.format(i)    
    actions = np.load(add1+add2+add3)
    act_size.append(np.shape(actions)[0])
    act_all += np.shape(actions)[0]


print('actino list:', act_size)
print('total number of actions:', act_all)