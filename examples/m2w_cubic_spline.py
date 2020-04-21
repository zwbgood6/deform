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
show_next_state = False
show_points = True
generate_video = True
video_dir = './video/'
video_name = 'line2w_spline.avi'

# test environment
for k in range(10):    
    ## generate following line
    #step1: select a random point in all points 
    if k < 4:
        index = 20
    elif k >= 4 and k < 7:
        index = 0
    else:    
        index = num_points - 1 
    # step2: generate a random moving angle
    if k < 4:
        move_angle = np.pi / 2 
    elif k >= 4 and k < 7:
        move_angle = 0.75 * np.pi
    else:
        move_angle = 0.25 * np.pi   
    #move_angle = np.pi/2 - (3 * k) * np.pi / 180  
    # step3: generate a random moving length
    move_length = 0.6
    # step4: generate new line position
    action = np.array([move_length, move_angle, index])
    rope.step(action)
    rope.render(image_dir=image_dir, spline=spline, show_points=show_points, show_next_state=show_next_state, generate_video=generate_video, video_dir=video_dir, video_name=video_name)

rope.close()




