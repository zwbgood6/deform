'''
learn from https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/5
'''

import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, image_paths_bi, image_paths_ori, resize_actions, actions):
        self.image_paths_bi = image_paths_bi # binary mask
        self.image_paths_ori = image_paths_ori # original
        self.resz_actions = resize_actions
        self.actions = actions

    def __getitem__(self, index):
        n = self.__len__()
        if index == n-1:
            index = index - 1
        # load images (pre-transform images)
        image_bi_pre = Image.open(self.image_paths_bi[index])
        image_bi_post = Image.open(self.image_paths_bi[index+1])
        image_bi_pre = self.transform_img(image_bi_pre)
        image_bi_post = self.transform_img(image_bi_post)
        image_ori_pre = plt.imread(self.image_paths_ori[index])
        image_ori_post = plt.imread(self.image_paths_ori[index+1])

        # load action (pre-transform x, y positions in action)
        action = self.actions[index]
        resz_action = self.resz_actions[index]
        #ratio = output_size / current_size
        #action = self.transform_act(action, ratio)

        '''
        sample = {state, action, next_state, 
                  state, action, next_state}
        image_bi_pre: 50*50, binary mask image pre-action
        resz_action: resized x,y in action
        image_bi_post: 50*50, binary mask image post-action
        image_ori_pre: 240*240, original image pre-action
        action: original action
        image_ori_post: 50*50, original image post-action        
        '''
        sample = {'image_bi_pre': image_bi_pre, 'resz_action': resz_action, 'image_bi_post': image_bi_post, 
                'image_ori_pre': image_ori_pre, 'action': action, 'image_ori_post': image_ori_post}
        return sample

    def __len__(self):
        return len(self.image_paths_bi)

    def transform_img(self, image):
        #image = TF.to_grayscale(image)  # TODO: change image to grayscale before traning
        #resize = transforms.Resize(size=(output_size, output_size)) # TODO: resize image before traning
        #image = resize(image)        
        image = TF.to_tensor(image) # pixel value range [0,1]
        # binarize image
        image = image > 0.3
        image = image.float()
        return image

    # def transform_act(self, action, ratio):
    #     action[:2] = action[:2] * ratio
    #     return action


def create_image_path(folder, total_img_num):
    '''create image_path list as input of MyDataset
    total_img_num: number of images 
    '''
    add1 = './rope_dataset/{}'.format(folder)
    image_paths = []
    for i in range(total_img_num):
        if len(str(i)) == 1:
            add2 = '/img_0000{}.jpg'.format(i)
        elif len(str(i)) == 2:
            add2 = '/img_000{}.jpg'.format(i)
        elif len(str(i)) == 3:
            add2 = '/img_00{}.jpg'.format(i)   
        elif len(str(i)) == 4:
            add2 = '/img_0{}.jpg'.format(i)    
        elif len(str(i)) == 5:
            add2 = '/img_{}.jpg'.format(i)                       
        image_paths.append(add1+add2)
    return image_paths

# def create_image_path(run_num, total_img_num):
#     '''create image_path list as input of MyDataset
#     run_num: string, e.g., 'run03'
#     total_img_num: number of images under 'run03'
#     '''
#     add1 = './rope_dataset/rope_seg/'
#     add2 = run_num
#     image_paths = []
#     for i in range(total_img_num):
#         if len(str(i)) == 1:
#             add3 = '/img_000{}.jpg'.format(i)
#         elif len(str(i)) == 2:
#             add3 = '/img_00{}.jpg'.format(i)
#         elif len(str(i)) == 3:
#             add3 = '/img_0{}.jpg'.format(i)   
#         elif len(str(i)) == 4:
#             add3 = '/img_{}.jpg'.format(i)                   
#         image_paths.append(add1+add2+add3)
#     return image_paths

# #pass unit test
# if __name__ == "__main__":
#     image_paths = create_image_path(run_num='run05', total_img_num=100)
#     dataset = MyDataset(image_paths)
#     n = dataset.__len__()
#     for i in range(n):
#         x = dataset.__getitem__(i)
#         x_ = x.view(-1, 128)
