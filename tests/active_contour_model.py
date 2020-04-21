# active contour model/snake
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

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

#img = data.astronaut()
#img = cv2.imread('/home/zwenbo/Documents/research/deform/rope_dataset/process_img/test1.jpeg')
img = cv2.imread('/home/zwenbo/Documents/research/deform/rope_dataset/rope/run03/img_0163.jpg')
img = rgb2gray(img)

#s = np.linspace(0, 2*np.pi, 400)
#r = 100 + 100*np.sin(s)
#c = 220 + 100*np.cos(s)
#np.concatenate()

# run03/img_0163.jpg
point_matrix = np.array([[118.3, 205.7],
                         [98.7, 152.7],
                         [148.1, 141.6],
                         [172.2, 130.5],
                         [177.5, 113.5],
                         [140, 95],
                         [129, 77],
                         [155, 75.6],
                         [161.5, 98.4],
                         [169, 110.8],
                         [195.8, 124.2],
                         [222.5, 143.8],
                         [237.6, 162]])

# img_0000
#r = np.hstack((np.linspace(26, 44, 30), np.linspace(44, 73, 30), np.linspace(73, 135, 30)))  
#c = np.hstack((np.linspace(15, 86, 30), np.linspace(86, 40, 30), np.linspace(40, 237, 30)))  

# img_0103
# point_matrix = np.array([[32, 96],
#                          [59, 147],
#                          [148, 105],
#                          [204, 108],
#                          [202, 117],
#                          [189, 123],
#                          [150, 121],
#                          [132, 142],
#                          [239, 167]])

#img 7.png
# point_matrix = np.array([[125, 80],
#                          [275, 152],
#                          [338, 69],
#                          [356, 64],
#                          [404, 235]])

# img test1.jpeg
# point_matrix = np.array([[521, 11],
#                          [509, 628],
#                          [294, 869],
#                          [389, 1048],
#                          [542, 892],
#                          [533, 628],
#                          [632, 498]])

r, c = generate_pos(point_matrix)
#r = np.hstack((np.linspace(97, 149, 30), np.linspace(149, 111, 30), np.linspace(111, 134, 30), np.linspace(134, 167, 30)))  
#c = np.hstack((np.linspace(34, 61, 30), np.linspace(61, 202, 30), np.linspace(202, 147, 30), np.linspace(147, 237, 30)))  
init = np.array([r, c]).T

snake = active_contour(gaussian(img, 1), init, boundary_condition='fixed',
                       alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1,
                       coordinates='rc')

plt.subplot(121)


plt.subplot(122)
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])

plt.show()