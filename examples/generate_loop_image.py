import numpy as np
import matplotlib.pyplot as plt
from deform.utils.utils import *
import cv2
import os

# image size
width, height = 4, 3

# rope initial point
x, y = 0.2, 0.5#float(np.random.random(1)), float(np.random.random(height))

# link length
d = 0.1

# collisoin check
# def collision_check():
#     return 
x_all = [x]
y_all = [y]

# generate initial points
num_points = 40
for i in range(num_points-1):
    phi = np.random.uniform(-np.pi/10, np.pi/10)
    #phi = np.random.uniform(0, np.pi/2)
    x1, y1 = x+d*np.cos(phi), y+d*np.sin(phi)
    x_all.append(x1)
    y_all.append(y1)
    x, y = x1, y1

# generate images
#images = []
#image_folder = '/home/zwenbo/Documents/research/deform/deform/img'
#video_name = '/home/zwenbo/Documents/research/deform/deform/video/video_line2knot_v1.avi'

for k in range(50):    
    ## generate following line
    # step1: select a random point in all points 
    if k < 30:
        index = 0
    else:
        index = 39      
    # step2: generate a random moving angle
    if k < 30:
        move_angle = np.pi/2 - (3 * k) * np.pi / 180
    elif k < 50:
        move_angle = (3 * k) * np.pi / 180     
    # step3: generate a random moving length
    move_length = 0.13
    # step4: generate new line position
    new_x_all, new_y_all = generate_new_line(x_all, y_all, index, move_angle, move_length, d, num_points)

    # plot the points
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.plot(new_x_all, new_y_all, c='b')
    action = get_action(move_angle, move_length)
    grip_pos_before = np.array([x_all[index], y_all[index]])
    grip_pos_after = get_pos_after(grip_pos_before, action)
    x_action_line = [grip_pos_before[0], grip_pos_after[0]]
    y_action_line = [grip_pos_before[1], grip_pos_after[1]]
    plt.scatter(x_action_line , y_action_line, c='r', s=5)
    dir = '/home/zwenbo/Documents/research/deform/deform/img/' + str(k) + '.png'
    plt.savefig(dir)
    plt.close()

    x_all = new_x_all
    y_all = new_y_all

#frame = cv2.imread(os.path.join(image_folder, images[0]))
#height, width, layers = 480, 680, 3#frame.shape
#video = cv2.VideoWriter(video_name, 0, 1, (width,height))

#for image in images:
#   video.write(cv2.imread(os.path.join(image_folder, image)))

#cv2.destroyAllWindows()
#video.release()

# # plot the points
# ax1 = plt.subplot(211)
# plt.xlim(0, width)
# plt.ylim(0, height)
# plt.plot(x_all,y_all, c='g')

# ax2 = plt.subplot(212)
# plt.xlim(0, width)
# plt.ylim(0, height)
# plt.plot(new_x_all,new_y_all, c='g')

# plt.show()