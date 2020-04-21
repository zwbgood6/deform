import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/home/zwenbo/Documents/research/deform/rope_dataset/rope/run03/img_0000.jpg')

# average
average_blur = cv2.blur(img,(5,5))

# gaussian
gaussian_blur = cv2.GaussianBlur(img,(5,5),0)

# median 
median = cv2.medianBlur(img,5)

# bilateral filter
bilateral_blur = cv2.bilateralFilter(img, 9, 150, 150)

# plt.subplot(151),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(152),plt.imshow(average_blur),plt.title('Average Blurred')
# plt.xticks([]), plt.yticks([])
# plt.subplot(153),plt.imshow(gaussian_blur),plt.title('Gaussian Blurred')
# plt.xticks([]), plt.yticks([])
# plt.subplot(154),plt.imshow(median),plt.title('Median')
# plt.xticks([]), plt.yticks([])
# plt.subplot(155),plt.imshow(bilateral_blur),plt.title('Bilateral Blurred')
# plt.xticks([]), plt.yticks([])
# plt.show()


plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(bilateral_blur),plt.title('Bilateral Blurred')
plt.xticks([]), plt.yticks([])
plt.show()