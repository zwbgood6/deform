from PIL import Image
import cv2
import os
import subprocess
import numpy as np

def add_mask(picture, width, height):
    # Process every pixel
    for x in range(0,width):
        for y in range(0,28):
            #current_color = picture.getpixel( (x,y) )
            new_color = (0, 255, 0)
            picture.putpixel( (x,y), new_color)

    for x in range(0,width):
        for y in range(190,height):
            new_color = (0, 255, 0)
            picture.putpixel( (x,y), new_color)

    for x in range(196, 240):
        a = int(85*x/44-357.6364)
        for y in range(21, a):
            new_color = (0, 255, 0) 
            picture.putpixel( (x,y), new_color)    
    return picture   

#num_range = np.linspace(6, 68, 63, dtype=int)
add1 = '/home/zwenbo/Documents/research/deform/rope_dataset/simplified_dataset'

#for j in num_range:
for j in range(60, 69):
    if len(str(j)) == 1:
        add2 = 'run0{}'.format(j)
    elif len(str(j)) == 2:
        add2 = 'run{}'.format(j)

    # count number of images in each folder
    add = 'cd ' + '/home/zwenbo/Documents/research/deform/rope_dataset/rope/' + add2 + '; ls -1 | wc -l'
    num_file = int(subprocess.check_output(add, shell=True)) - 1

    for i in range(num_file):
        
        if not os.path.exists(add1 + '/rope_no_loop_green_mask/' + add2):
            os.mkdir(add1 + '/rope_no_loop_green_mask/' + add2)
        if len(str(i)) == 1:
            add3 = '/img_000{}.jpg'.format(i)
        elif len(str(i)) == 2:
            add3 = '/img_00{}.jpg'.format(i)
        elif len(str(i)) == 3:
            add3 = '/img_0{}.jpg'.format(i) 
        elif len(str(i)) == 4:
            add3 = '/img_{}.jpg'.format(i) 
        if not os.path.exists(add1 + '/rope_no_loop/' + add2 + add3):
            continue
        picture = Image.open(add1 + '/rope_no_loop/' + add2 + add3)
        width, height = picture.size
        picture = add_mask(picture, width, height)
        picture.save(add1 + '/rope_no_loop_green_mask/' + add2 + add3)
# picture = Image.open("./rope_dataset/rope/run05/1.jpg")
# width, height = picture.size



#picture.save('./rope_dataset/rope/run05/2.jpg')
