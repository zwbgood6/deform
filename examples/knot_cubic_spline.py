import numpy as np
from deform.envs.rope import RopeEnv
from deform.utils.params import *
from deform.utils.utils import *

# generate initial points
x_all, y_all = generate_initial_points(x_init, y_init, num_points, link_length)
#x_all, y_all = np.array(x_all), np.array(y_all)
start_state = np.column_stack((x_all, y_all))

# create environment 
rope = RopeEnv(start_state, link_length, num_points)
image_dir = './img/'
spline = True
show_points=True 
show_next_state = True
generate_video = True
video_dir = './video/'
video_name = 'knot_spline.avi'

# test environment
for k in range(63):    
    ## generate following line
    #step1: select a random point in all points 
    if k < 30:
        index = 0
    else:
        index = num_points - 1
    #index = 0    
    # step2: generate a random moving angle
    if k < 30:
        move_angle = np.pi/2 - (3 * k) * np.pi / 180
    elif k < 50:
        move_angle = (3 * k) * np.pi / 180
    else:
        move_angle = -np.pi/2 + 7 * (k - 50) * np.pi / 180  
    #move_angle = np.pi/2 - (3 * k) * np.pi / 180  
    # step3: generate a random moving length
    move_length = 0.13
    # step4: generate new line position
    action = np.array([move_length, move_angle, index])
    rope.step(action)
    rope.render(image_dir=image_dir, spline=spline, show_points=show_points, show_next_state=show_next_state, generate_video=generate_video, video_dir=video_dir, video_name=video_name)

rope.close()




