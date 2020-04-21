# 1. Thresholding the image based on color range
# 2. Apply active contour model/snake

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from sort import *
from scipy import interpolate

def seg_img(image):
    '''segment image based on green color in HSV space
    image: color image
    rope: gray scale rope image
    '''    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # light green
    lower_green_light = np.array([20,20,50])
    upper_green_light = np.array([130,150,255])
    # normal green
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([70, 255, 255])

    # Threshold with inRange() get only specific colors
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_green_light = cv2.inRange(hsv, lower_green_light, upper_green_light)

    # union two masks
    mask_union = cv2.bitwise_or(mask_green, mask_green_light)

    rope = np.where(mask_union==0, mask_union, 167)
    rope = rgb2gray(rope)
    
    return rope

def generate_pos(point_matrix):
    '''generate r and c for piecewise linear lines
    point_matrix: n*2 array
    '''
    n = np.shape(point_matrix)[0]
    for i in range(n-1):
        if i == 0:
            r = np.linspace(point_matrix[i][1], point_matrix[i+1][1])
            c = np.linspace(point_matrix[i][0], point_matrix[i+1][0])
        else:
            r = np.hstack((r, np.linspace(point_matrix[i][1], point_matrix[i+1][1])))   
            c = np.hstack((c, np.linspace(point_matrix[i][0], point_matrix[i+1][0])))  

    return r, c

def select_points(points, number):
    '''select number of points in points array
    points: Nx2 array
    number: how many points to select out of N
    '''
    num_point = np.shape(points)[0]
    if num_point > number:
        interval = int(num_point/number)
        idx = np.arange(0, interval*number, interval)
        points = points[idx][:]
    return points

def sort_point_pos(x, y):
    idx = np.argsort(x)
    x = x.tolist()
    x.sort()
    new_x = np.array(x) 
    new_y = y[idx]
    return new_x, new_y

for i in range(234):

    # load image
    add1 = '/home/zwenbo/Documents/research/deform/rope_dataset'
    add2_load = '/rope_crop'
    add2_save = '/rope_snake'
    if len(str(i)) == 1:
        add3_load = '/run03/img_000{}.jpg'.format(i)
        add3_save = '/run03/img_000{}.png'.format(i)
    elif len(str(i)) == 2:
        add3_load = '/run03/img_00{}.jpg'.format(i)
        add3_save = '/run03/img_00{}.png'.format(i)
    elif len(str(i)) == 3:
        add3_load = '/run03/img_0{}.jpg'.format(i)
        add3_save = '/run03/img_0{}.png'.format(i)                            
    add_load = add1 + add2_load + add3_load
    add_save = add1 + add2_save + add3_save
    img = cv2.imread(add_load)

    # segment image
    rope = seg_img(img)

    # ## snake
    # # prepare initial points
    i, j = np.where(rope==0)
    point_matrix = np.c_[j, i] 
    point_matrix = select_points(point_matrix, 18)

    T = create_graph(point_matrix)
    _, opt_order = find_start_node(T, point_matrix)
    x, y = point_matrix[:,0], point_matrix[:,1]
    xx1 = x[opt_order]
    yy1 = y[opt_order]  

    point_matrix = np.c_[xx1, yy1]
    r, c = generate_pos(point_matrix)
    init = np.array([r, c]).T

    # # snake
    snake = active_contour(gaussian(rope, 1.6), init, boundary_condition='fixed',
                        alpha=1, beta=100.0, w_line=-5, w_edge=0, gamma=0.1,
                        coordinates='rc')
    
    # plot
    fig, ax1 = plt.subplots()
    ax1.imshow(gaussian(rope, 1.6), cmap=plt.cm.gray)
    ax1.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax1.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax1.set_xticks([]), ax1.set_yticks([])
    plt.savefig(add_save)
    plt.close()
    #plt.show()


    ## interp 1D
    # prepare initial points
    # i, j = np.where(rope==0)
    # point_matrix = np.c_[j, i] 
    # point_matrix = select_points(point_matrix, 40)
    # x, y = point_matrix[:,0], point_matrix[:,1]
    # sort_x, sort_y = sort_point_pos(x, y)
    # f = interpolate.interp1d(sort_x, sort_y, kind='slinear')
    # minimum = max(0, min(x))
    # xnew = np.arange(minimum, max(x), 0.1)
    # ynew = f(xnew)   # use interpolation function returned by `interp1d`
    # fig, ax1 = plt.subplots()
    # ax1.imshow(gaussian(rope, 1.6), cmap=plt.cm.gray)
    # plt.plot(x, y, 'o', xnew, ynew, '-')
    # plt.show()





