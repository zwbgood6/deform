from deform.model.create_dataset import *
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from deform.utils.utils import *

def show_landmarks(image_bi_pre, image_bi_post, resz_action):
    """Show image with landmarks"""
    plt.imshow(image_bi_pre)
    rect(resz_action[:4], 'blue')
    #plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

print('***** Preparing Data *****')
total_img_num = 77944
train_num = int(total_img_num * 0.8)
image_paths_bi = create_image_path('rope_all_resize_gray', total_img_num)
resz_act_path = './rope_dataset/rope_all_resize_gray/resize_actions.npy'
resz_act = np.load(resz_act_path)
dataset = MyDataset(image_paths_bi, resz_act)

# transformation setting
trans = Translation(10)
hflip = HFlip(1)
vflip = VFlip(1)
composed = transforms.Compose([Translation(2),
                               HFlip(1)])

# plot figure
fig = plt.figure()
sample = dataset[1355]
tsfrm = trans
transformed_sample = tsfrm(sample)

ax = plt.subplot(1, 2, 1)
plt.tight_layout()
ax.set_title('Original')
show_landmarks(**sample)

ax = plt.subplot(1, 2, 2)
plt.tight_layout()
ax.set_title(type(tsfrm).__name__)
show_landmarks(**transformed_sample)

#
# for i, tsfrm in enumerate([trans, hflip, vflip]):
#     transformed_sample = tsfrm(sample)
#     ax = plt.subplot(1, 3, i + 1)
#     plt.tight_layout()
#     ax.set_title(type(tsfrm).__name__)
#     show_landmarks(**transformed_sample)

plt.show()