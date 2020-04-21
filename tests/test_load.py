import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

train_img = []
for i in range(10):
    image_path = './img/' + str(i) + '.png'
    img = imread(image_path, as_gray=True)
    img /= 255.0
    img = img.astype('float32')
    train_img.append(img)

train_x = np.array(train_img)

plt.figure(figsize=(10, 10))
plt.subplot(221), plt.imshow(train_x[0], cmap='gray') 
plt.subplot(222), plt.imshow(train_x[1], cmap='gray') 
plt.subplot(223), plt.imshow(train_x[2], cmap='gray') 
plt.subplot(224), plt.imshow(train_x[3], cmap='gray') 

plt.show()