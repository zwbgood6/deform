import cv2
import matplotlib.pyplot as plt

## rope parameters
# rope initial point
x_init, y_init = 0.2, 1
# segment length
link_length = 0.1
# default number of points on a rope
num_points = 40

## render parameters
# screen size
screen_width, screen_height = 800, 600