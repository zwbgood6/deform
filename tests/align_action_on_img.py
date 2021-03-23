import numpy as np                   
import math
import matplotlib.pyplot as plt
import os
import cv2

def rect(poke, c, label=None):
    # from rope.ipynb in Berkeley's rope dataset file
    x, y, t, l = poke
    dx = -100 * l * math.cos(t)
    dy = -100 * l * math.sin(t)
    arrow = plt.arrow(x, y, dx, dy, width=0.001, head_width=6, head_length=6, color=c, label=label)        
    #plt.legend([arrow,], ['My label',])

def plot_action(resz_action, recon_action, directory):
    # from rope.ipynb in Berkeley's rope dataset file
    plt.figure()
    # upper row original
    plt.subplot(1, 2, 1)
    rect(resz_action[i], "blue")
    plt.axis('off') 
    # middle row reconstruction
    plt.subplot(1, 2, 2)
    rect(recon_action[i], "red")
    plt.axis('off')
    plt.savefig(directory) 
    plt.close()

def plot_sample(img_before, img_after, resz_action, recon_action, directory):
    # from rope.ipynb in Berkeley's rope dataset file
    plt.figure()
    N = int(img_before.shape[0])
    for i in range(N):
        # upper row original
        plt.subplot(3, N, i+1)
        rect(resz_action[i], "blue")
        plt.imshow(img_before[i].reshape((50,50)))
        plt.axis('off') 
        # middle row reconstruction
        plt.subplot(3, N, i+1+N)
        rect(recon_action[i], "red")
        plt.imshow(img_before[i].reshape((50,50)))
        plt.axis('off')
        # lower row: next image after action
        plt.subplot(3, N, i+1+2*N)
        plt.imshow(img_after[i].reshape((50,50)))
        plt.axis('off')
    plt.savefig(directory) 
    plt.close()

action = np.load("/home/zwenbo/Documents/research/deform/rope_dataset/rope_clean_all_resize50/rope_clean_all_size50.npy")
img = cv2.imread("/home/zwenbo/Documents/research/deform/rope_dataset/rope_clean_all_resize50/img_00100.jpg")
plt.figure()
plt.subplot(1, 1, 1)
rect(action[100][:4], "blue")
plt.imshow(img)
plt.axis('off') 
plt.show()    