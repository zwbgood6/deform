import numpy as np
import math

# a = [1, 2]
# b = [3, 1]
# theta1 = math.atan2(2-1, 1-3) # a-b
# theta2 = math.atan2(1-2, 3-1) # b-a
# print(theta1, theta2)
# num_points=40
# #index = np.random.randint(num_points)
# #print(index)

# move_angle = np.pi * np.random.random()
# print(move_angle)
#gripper_x_pos = 2.0
#x_all = np.array([1.,2.,3.])
#print(sum(gripper_x_pos>=x_all))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()


def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(1):
    x += np.pi / 15.
    y += np.pi / 20.
    im = plt.imshow(f(x, y), animated=True)
    #ims.append([im])

# ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
#                                 repeat_delay=1000)

# ani.save('dynamic_images.mp4')

plt.show()