import numpy as np
import glob, os
import shutil

src_dir = "/home/zwenbo/Documents/research/deform/rope_dataset/rope_clean"

count = 0
for i in range(3, 63):# 64,68
    if len(str(i)) == 1:
        folder_name = "/run0" + str(i)
    elif len(str(i)) == 2:
        folder_name = "/run" + str(i)
    # filelist = glob.glob(src_dir + folder_name +"/*.jpg")   
    # count = len(filelist)
    for j in range(9999):
        if len(str(j)) == 1:
            file_name = "/img_000" + str(j) + ".jpg"
        elif len(str(j)) == 2:
            file_name = "/img_00" + str(j) + ".jpg"
        elif len(str(j)) == 3:
            file_name = "/img_0" + str(j) + ".jpg"
        elif len(str(j)) == 4:
            file_name = "/img_" + str(j) + ".jpg"    
        if os.path.exists(src_dir+folder_name+file_name):
            newPath = shutil.copy(med_dir+folder_name+file_name, tar_dir+folder_name+file_name)
        else:
            continue    
