import numpy as np

add1 = './rope_dataset/rope/'
add2 = 'run'
add3 = '/actions.npy'


bool_size = []
bool_all = 0
for i in range(3, 69):
    if len(str(i)) == 1:
        add2 = 'run0{}'.format(i)
    elif len(str(i)) == 2:
        add2 = 'run{}'.format(i)    
    actions = np.load(add1+add2+add3)
    boolean = actions[-1][-1] 
    bool_size.append(boolean)
    bool_all += boolean
    #np.save(add1+add2+add3, actions)


print('bool list:', bool_size)
print('total number of bools:', bool_all)