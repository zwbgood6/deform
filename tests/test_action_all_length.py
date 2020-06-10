import numpy as np

actions = np.load('/home/zwenbo/Documents/research/deform/rope_dataset/rope_all_resize_gray/resize_actions.npy')
max_x = 0
max_y = 0
max_angle = 0
max_l = 0
min_x = 50
min_y = 50
min_angle = 50
min_l = 50
for action in actions:
    if action[4] == 1:
        if action[0] < min_x:
            min_x = action[0]
        if action[0] > max_x:
            max_x = action[0]

        if action[1] < min_y:
            min_y = action[1]
        if action[1] > max_y:
            max_y = action[1]   

        if action[2] < min_angle:
            min_angle = action[2]
        if action[2] > max_angle:
            max_angle = action[2]

        if action[3] < min_l:
            min_l = action[3]
        if action[3] > max_l:
            max_l = action[3]                 

print('max x:', max_x)
print('min x:', min_x)
print('____________')
print('max y:', max_y)
print('min y:', min_y)
print('____________')
print('max angle:', max_angle)
print('min angle:', min_angle)
print('____________')
print('max l:', max_l)
print('min l:', min_l)