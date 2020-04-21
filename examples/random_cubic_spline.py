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
show_next_state = False
generate_video = True
video_dir = './video/'
video_name = 'random_spline.avi'

# test environment
for k in range(50):    
    ## generate following line
    # step1: select a random point in all points 
    index = int(np.random.randint(num_points))
    # step2: generate a random moving angle
    move_angle = np.pi * np.random.rand()
    # step3: generate a random moving length
    max_move_length = 0.6
    move_length = max_move_length * np.random.rand()
    # step4: generate new line position
    action = np.array([move_length, move_angle, index])
    rope.step(action)
    rope.render(image_dir=image_dir, spline=spline, show_points=show_points, show_next_state=show_next_state, generate_video=generate_video, video_dir=video_dir, video_name=video_name)

rope.close()




