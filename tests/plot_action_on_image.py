from deform.model.create_dataset import *
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from deform.utils.utils import *

def show_landmarks(image, action):
    """Show image with landmarks"""
    plt.imshow(image)
    if action is not None:
        rect(action[:4], 'blue')
    #plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

print('***** Preparing Data *****')
total_img_num = 22515
train_num = int(total_img_num * 0.8)
image_paths_bi = create_image_path('rope_no_loop_all_resize_gray', total_img_num)
resz_act_path = './rope_dataset/rope_no_loop_all_resize_gray/resize_actions.npy'
resz_act = np.load(resz_act_path)
dataset = MyDataset(image_paths_bi, resz_act)

image_paths_bi_ori = create_image_path('rope_no_loop_all_gray', total_img_num)
resz_act_path_ori = './rope_dataset/rope_no_loop_all_gray/actions.npy'
resz_act_ori = np.load(resz_act_path_ori)
dataset_ori = MyDataset(image_paths_bi_ori, resz_act_ori)

# plot figure
fig = plt.figure()
sample = dataset[1355]
image_bi_pre = sample['image_bi_pre']
image_bi_cur = sample['image_bi_cur']
image_bi_post = sample['image_bi_post']
resz_action_pre = sample['resz_action_pre']
resz_action_cur = sample['resz_action_cur']

sample_ori = dataset_ori[1355]
image_bi_pre_ori = sample_ori['image_bi_pre']
image_bi_cur_ori = sample_ori['image_bi_cur']
image_bi_post_ori = sample_ori['image_bi_post']
resz_action_pre_ori = sample_ori['resz_action_pre']
resz_action_cur_ori = sample_ori['resz_action_cur']

ax = plt.subplot(2, 3, 1)
plt.tight_layout()
ax.set_title('Previous')
show_landmarks(image_bi_pre, resz_action_pre)

ax = plt.subplot(2, 3, 2)
plt.tight_layout()
ax.set_title('Current')
show_landmarks(image_bi_cur, resz_action_cur)

ax = plt.subplot(2, 3, 3)
plt.tight_layout()
ax.set_title('Post')
show_landmarks(image_bi_post, None)

ax = plt.subplot(2, 3, 4)
plt.tight_layout()
ax.set_title('Previous')
show_landmarks(image_bi_pre_ori, resz_action_pre_ori)

ax = plt.subplot(2, 3, 5)
plt.tight_layout()
ax.set_title('Current')
show_landmarks(image_bi_cur_ori, resz_action_cur_ori)

ax = plt.subplot(2, 3, 6)
plt.tight_layout()
ax.set_title('Post')
show_landmarks(image_bi_post_ori, None)

plt.show()