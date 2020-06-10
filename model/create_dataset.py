import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import random
import math
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataloader import default_collate

class MyDataset(Dataset):
    '''
    learn from https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/5
    '''    
    def __init__(self, image_paths_bi, resize_actions, transform=None):
        self.image_paths_bi = image_paths_bi # binary mask
        #self.image_paths_ori = image_paths_ori # original
        self.resz_actions = resize_actions
        #self.actions = actions
        self.transform = transform

    def __getitem__(self, index):
        # try:
        #     return super(MyDataset, self).__getitem__(index)
        # except Exception as e:
        #     print(e)    
        n = self.__len__()
        if index == n-1:
            index = index - 1

        # load action 
        resz_action = self.resz_actions[index]
        # decide if action is valid
        if resz_action[4] == 0:
            return {'image_bi_pre': None, 'image_bi_post': None, 'resz_action': None}

        # load images (pre-transform images)
        image_bi_pre = Image.open(self.image_paths_bi[index])
        image_bi_post = Image.open(self.image_paths_bi[index+1])
        #image_bi_pre = self.transform_img(image_bi_pre, trans)
        #image_bi_post = self.transform_img(image_bi_post, trans)
        #image_ori_pre = plt.imread(self.image_paths_ori[index])
        #image_ori_post = plt.imread(self.image_paths_ori[index+1])

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
        # sample = {'image_bi_pre': image_bi_pre, 'resz_action': resz_action, 'image_bi_post': image_bi_post, 
        #         'image_ori_pre': image_ori_pre, 'action': action, 'image_ori_post': image_ori_post}
        sample = {'image_bi_pre': image_bi_pre, 'image_bi_post': image_bi_post, 'resz_action': resz_action[:4]}       

        # random transformation in [-2,2]
        #trans = None  
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_paths_bi)

    # def transform_img(self, image, trans):    
    #     image = TF.to_tensor(image) # pixel value range [0,1]
    #     if self.transform:
    #         image = TF.affine(image, angle=0, translate=trans, scale=0.0, shear=0.0)
    #     # binarize image
    #     image = image > 0.3
    #     image = image.float()
    #     return image

    # def transform_act(self, action, trans):
    #     # transform based on trans
    #     if self.transform:
    #         action[:2] = action[:2] * np.array(trans)
    #     # get first four elements in the action
    #     return action[:4]

class Translation(object):
    '''Translate the image and action [-max_translation, max_translation]. e.g., [-10, 10]
    '''
    def __init__(self, max_translation=10):
        # max_translation x in [2.5, 44.58], y in [2.5, 33.54]
        self.m_trans = max_translation


    def __call__(self, sample):
        image_bi_pre, image_bi_post, action = sample['image_bi_pre'], sample['image_bi_post'], sample['resz_action']
        trans = list(2 * self.m_trans * np.random.random_sample((2,)) - self.m_trans)
        trans_action = action.copy()
        trans_action[:2] = trans_action[:2] + np.array(trans)
        image_bi_pre = TF.affine(image_bi_pre, angle=0, translate=trans, scale=1.0, shear=0.0)
        image_bi_post = TF.affine(image_bi_post, angle=0, translate=trans, scale=1.0, shear=0.0)

        return {'image_bi_pre': image_bi_pre, 'image_bi_post': image_bi_post, 'resz_action': trans_action}

# class Rotation(object):
#     '''Rotate the image by certain angle [-max_angle, max_angle], e.g., [-pi/4, pi/4]
#     '''
#     def __init__(self, max_angle):
#         self.m_angle = max_angle

#     def __call__(self, sample):
#         image_bi_pre, image_bi_post, resz_action = sample['image_bi_pre'], sample['image_bi_post'], sample['resz_action']
#         rot = 2 * self.m_angle * np.random.random_sample() - self.m_angle
#         image_bi_pre = TF.rotate(image_bi_pre, angle=rot) # clockwise - negative, counter clockwise - positive
#         image_bi_post = TF.rotate(image_bi_post, angle=rot)
#         # TODO: rotate action; cannot rotate the action because it's in world frame
#         return None
#         #return {'image_bi_pre': image_bi_pre, 'image_bi_post': image_bi_post, 'resz_action': resz_action}

class HFlip(object):
    '''Ramdom Horizontal flip the image and action with probabilty p
    '''
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        else:    
            image_bi_pre, image_bi_post, action = sample['image_bi_pre'], sample['image_bi_post'], sample['resz_action']
            image_bi_pre, image_bi_post = TF.hflip(image_bi_pre), TF.hflip(image_bi_post)
            # position: x=50-x, y=y
            tsfrm_action = action.copy()
            tsfrm_action[0] = image_bi_pre.size[0] - tsfrm_action[0]
            # angle in [0, 2*pi]
            if tsfrm_action[2] > math.pi:
                tsfrm_action[2] = 3 * math.pi - tsfrm_action[2] # angle=3*pi-angle
            else:
                tsfrm_action[2] = math.pi - tsfrm_action[2] # angle=pi-angle 
            return {'image_bi_pre': image_bi_pre, 'image_bi_post': image_bi_post, 'resz_action': tsfrm_action}

class VFlip(object):
    '''Ramdom vertical flip the image and action with probabilty p
    '''
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        else:    
            image_bi_pre, image_bi_post, action = sample['image_bi_pre'], sample['image_bi_post'], sample['resz_action']
            image_bi_pre, image_bi_post = TF.vflip(image_bi_pre), TF.vflip(image_bi_post)
            # position: x=x, y=50-y
            tsfrm_action = action.copy()
            tsfrm_action[1] = image_bi_pre.size[1] - tsfrm_action[1]
            # angle in [0, 2*pi], angle=2*pi-angle 
            tsfrm_action[2] = 2 * math.pi - tsfrm_action[2] 
            return {'image_bi_pre': image_bi_pre, 'image_bi_post': image_bi_post, 'resz_action': tsfrm_action}

class ToTensor(object):
    '''convert ndarrays in sample to tensors
    '''
    def __call__(self, sample):
        image_bi_pre, image_bi_post, resz_action = sample['image_bi_pre'], sample['image_bi_post'], sample['resz_action']
        # to tensor and binarize image
        image_bi_pre = TF.to_tensor(image_bi_pre) > 0.3
        image_bi_post = TF.to_tensor(image_bi_post) > 0.3
        
        return {'image_bi_pre': image_bi_pre.float(), 'image_bi_post': image_bi_post.float(), 'resz_action': torch.tensor(resz_action)}

def my_collate(batch):
    '''filer out the data when sample['image_bi_post']=None, which means action's last element is zero
    from https://discuss.pytorch.org/t/questions-about-dataloader-and-dataset/806/8
    '''
    batch = list(filter(lambda x: x['image_bi_post'] is not None, batch))
    return default_collate(batch)

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
