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
        none_sample = {'image_bi_pre': None, 'image_bi_cur': None, 'image_bi_post': None, 'resz_action_pre': None, 'resz_action_cur': None}
        # edge cases, use index in [1,n-1) 
        if index == 0:
            index = np.random.randint(1, n-1)
        if index == n-1:
            index = np.random.randint(1, n-1)

        # load action 
        resz_action_pre = self.resz_actions[index-1]
        resz_action_cur = self.resz_actions[index]
        # decide if action is valid
        if int(resz_action_pre[4]) == 0 or int(resz_action_cur[4]) == 0:
            return none_sample

        # load images (pre-transform images)        
        image_bi_pre = Image.open(self.image_paths_bi[index-1])
        image_bi_cur = Image.open(self.image_paths_bi[index])        
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
        sample = {'image_bi_pre': image_bi_pre, 'image_bi_cur': image_bi_cur, 'image_bi_post': image_bi_post, 'resz_action_pre': resz_action_pre[:4], 'resz_action_cur': resz_action_cur[:4]}       

        # random transformation in [-2,2]
        #trans = None  
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_paths_bi)

class MyDatasetMultiPred4(Dataset):
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
        none_sample = {'image_bi_pre': None, 'image_bi_cur': None, 'image_bi_post': None, 'image_bi_post2': None, 'image_bi_post3': None,\
            'resz_action_pre': None, 'resz_action_cur': None, 'resz_action_post': None, 'resz_action_post2': None}
        # edge cases, use index in [1,n-1) 
        if index == 0:
            index = np.random.randint(1, n-3)
        if index == n-1:
            index = np.random.randint(1, n-3)
        if index == n-2:
            index = np.random.randint(1, n-3)                        
        if index == n-3:
            index = np.random.randint(1, n-3)

        # load action 
        resz_action_pre = self.resz_actions[index-1]
        resz_action_cur = self.resz_actions[index]
        resz_action_post = self.resz_actions[index+1]
        resz_action_post2 = self.resz_actions[index+2]
        # decide if action is valid
        if int(resz_action_pre[4]) == 0 or int(resz_action_cur[4]) == 0 or int(resz_action_post[4]) == 0 or int(resz_action_post2[4]) == 0:
            return none_sample

        # load images (pre-transform images)        
        image_bi_pre = Image.open(self.image_paths_bi[index-1])
        image_bi_cur = Image.open(self.image_paths_bi[index])        
        image_bi_post = Image.open(self.image_paths_bi[index+1])
        image_bi_post2 = Image.open(self.image_paths_bi[index+2])
        image_bi_post3 = Image.open(self.image_paths_bi[index+3])
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
        sample = {'image_bi_pre': image_bi_pre, 'image_bi_cur': image_bi_cur, 'image_bi_post': image_bi_post, 'image_bi_post2': image_bi_post2, 'image_bi_post3': image_bi_post3,\
            'resz_action_pre': resz_action_pre[:4], 'resz_action_cur': resz_action_cur[:4], 'resz_action_post': resz_action_post[:4], 'resz_action_post2': resz_action_post2[:4]}       

        # random transformation in [-2,2]
        #trans = None  
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_paths_bi)

class MyDatasetMultiPred10(Dataset):
    '''
    learn from https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/5
    '''    
    def __init__(self, image_paths_bi, resize_actions, transform=None):
        self.image_paths_bi = image_paths_bi # binary mask
        self.resz_actions = resize_actions
        self.transform = transform

    def __getitem__(self, index):   
        n = self.__len__()
        none_sample = {'image_bi_pre': None, 'image_bi_cur': None, 'image_bi_post': None, 'image_bi_post2': None, 'image_bi_post3': None,\
            'image_bi_post4': None, 'image_bi_post5': None, 'image_bi_post6': None, 'image_bi_post7': None, 'image_bi_post8': None, 'image_bi_post9': None,\
            'resz_action_pre': None, 'resz_action_cur': None, 'resz_action_post': None, 'resz_action_post2': None, 'resz_action_post3': None, 'resz_action_post4': None,\
            'resz_action_post5': None, 'resz_action_post6': None, 'resz_action_post7': None, 'resz_action_post8': None}
        # edge cases, use index in [1,n-1) 
        if index == 0:
            index = np.random.randint(1, n-9)
        if index == n-1:
            index = np.random.randint(1, n-9)
        if index == n-2:
            index = np.random.randint(1, n-9)                        
        if index == n-3:
            index = np.random.randint(1, n-9)
        if index == n-4:
            index = np.random.randint(1, n-9)
        if index == n-5:
            index = np.random.randint(1, n-9)
        if index == n-6:
            index = np.random.randint(1, n-9)
        if index == n-7:
            index = np.random.randint(1, n-9)
        if index == n-8:
            index = np.random.randint(1, n-9) 
        if index == n-9:
            index = np.random.randint(1, n-9)                                                            
        # load action 
        resz_action_pre = self.resz_actions[index-1]
        resz_action_cur = self.resz_actions[index]
        resz_action_post = self.resz_actions[index+1]
        resz_action_post2 = self.resz_actions[index+2]
        resz_action_post3 = self.resz_actions[index+3]
        resz_action_post4 = self.resz_actions[index+4]
        resz_action_post5 = self.resz_actions[index+5]
        resz_action_post6 = self.resz_actions[index+6]
        resz_action_post7 = self.resz_actions[index+7]
        resz_action_post8 = self.resz_actions[index+8]
        resz_action_post9 = self.resz_actions[index+9]                                                        
        # decide if action is valid
        if int(resz_action_pre[4]) == 0 or int(resz_action_cur[4]) == 0 \
            or int(resz_action_post[4]) == 0 or int(resz_action_post2[4]) == 0 \
            or int(resz_action_post3[4]) == 0 or int(resz_action_post4[4]) == 0 \
            or int(resz_action_post5[4]) == 0 or int(resz_action_post6[4]) == 0 \
            or int(resz_action_post7[4]) == 0 or int(resz_action_post8[4]) == 0 \
            or int(resz_action_post9[4]) == 0:
            return none_sample

        # load images (pre-transform images)        
        image_bi_pre = Image.open(self.image_paths_bi[index-1])
        image_bi_cur = Image.open(self.image_paths_bi[index])        
        image_bi_post = Image.open(self.image_paths_bi[index+1])
        image_bi_post2 = Image.open(self.image_paths_bi[index+2])
        image_bi_post3 = Image.open(self.image_paths_bi[index+3])
        image_bi_post4 = Image.open(self.image_paths_bi[index+4])
        image_bi_post5 = Image.open(self.image_paths_bi[index+5])
        image_bi_post6 = Image.open(self.image_paths_bi[index+6])
        image_bi_post7 = Image.open(self.image_paths_bi[index+7])
        image_bi_post8 = Image.open(self.image_paths_bi[index+8])
        image_bi_post9 = Image.open(self.image_paths_bi[index+9])                                                
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
        sample = {'image_bi_pre': image_bi_pre, 'image_bi_cur': image_bi_cur, 'image_bi_post': image_bi_post, 'image_bi_post2': image_bi_post2, 'image_bi_post3': image_bi_post3,\
            'image_bi_post4': image_bi_post4, 'image_bi_post5': image_bi_post5, 'image_bi_post6': image_bi_post6, 'image_bi_post7': image_bi_post7, 'image_bi_post8': image_bi_post8, \
            'image_bi_post9': image_bi_post9, \
            'resz_action_pre': resz_action_pre[:4], 'resz_action_cur': resz_action_cur[:4], 'resz_action_post': resz_action_post[:4], 'resz_action_post2': resz_action_post2[:4], \
            'resz_action_post3': resz_action_post3[:4], 'resz_action_post4': resz_action_post4[:4], 'resz_action_post5': resz_action_post5[:4], 'resz_action_post6': resz_action_post6[:4], \
            'resz_action_post7': resz_action_post7[:4], 'resz_action_post8': resz_action_post8[:4]}
        # random transformation in [-2,2]
        #trans = None  
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_paths_bi)

class Translation(object):
    '''Translate the image and action [-max_translation, max_translation]. e.g., [-10, 10]
    '''
    def __init__(self, max_translation=10):
        # max_translation x in [2.5, 44.58], y in [2.5, 33.54]
        self.m_trans = max_translation


    def __call__(self, sample):
        image_bi_pre, image_bi_cur, image_bi_post, action_pre, action_cur = sample['image_bi_pre'], sample['image_bi_cur'], sample['image_bi_post'], sample['resz_action_pre'], sample['resz_action_cur']
        trans = list(2 * self.m_trans * np.random.random_sample((2,)) - self.m_trans)
        trans_action_pre, trans_action_cur = action_pre.copy(), action_cur.copy()
        trans_action_pre[:2], trans_action_cur[:2] = trans_action_pre[:2] + np.array(trans), trans_action_cur[:2] + np.array(trans)
        image_bi_pre = TF.affine(image_bi_pre, angle=0, translate=trans, scale=1.0, shear=0.0)
        image_bi_cur = TF.affine(image_bi_cur, angle=0, translate=trans, scale=1.0, shear=0.0)
        image_bi_post = TF.affine(image_bi_post, angle=0, translate=trans, scale=1.0, shear=0.0)

        return {'image_bi_pre': image_bi_pre, 'image_bi_cur': image_bi_cur, 'image_bi_post': image_bi_post, 'resz_action_pre': trans_action_pre, 'resz_action_cur': trans_action_cur}

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

    def flip_angle(self, action):
        # angle in [0, 2*pi]
        if action[2] > math.pi:
            action[2] = 3 * math.pi - action[2] # angle=3*pi-angle
        else:
            action[2] = math.pi - action[2] # angle=pi-angle         
        return action

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        else:    
            image_bi_pre, image_bi_cur, image_bi_post, action_pre, action_cur = sample['image_bi_pre'], sample['image_bi_cur'], sample['image_bi_post'], sample['resz_action_pre'], sample['resz_action_cur']
            image_bi_pre, image_bi_cur, image_bi_post = TF.hflip(image_bi_pre), TF.hflip(image_bi_cur), TF.hflip(image_bi_post)
            # position: x=50-x, y=y
            tsfrm_action_pre, tsfrm_action_cur = action_pre.copy(), action_cur.copy()
            tsfrm_action_pre[0], tsfrm_action_cur[0] = image_bi_pre.size[0] - tsfrm_action_pre[0], image_bi_cur.size[0] - tsfrm_action_cur[0]
            # angle in [0, 2*pi]
            tsfrm_action_pre = self.flip_angle(tsfrm_action_pre)
            tsfrm_action_cur = self.flip_angle(tsfrm_action_cur)

            return {'image_bi_pre': image_bi_pre, 'image_bi_cur': image_bi_cur, 'image_bi_post': image_bi_post, 'resz_action_pre': tsfrm_action_pre, 'resz_action_cur': tsfrm_action_cur}


class VFlip(object):
    '''Ramdom vertical flip the image and action with probabilty p
    '''
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        else:    
            image_bi_pre, image_bi_cur, image_bi_post, action_pre, action_cur = sample['image_bi_pre'], sample['image_bi_cur'], sample['image_bi_post'], sample['resz_action_pre'], sample['resz_action_cur']
            image_bi_pre, image_bi_cur, image_bi_post = TF.vflip(image_bi_pre), TF.vflip(image_bi_cur), TF.vflip(image_bi_post)
            # position: x=x, y=50-y
            tsfrm_action_pre, tsfrm_action_cur = action_pre.copy(), action_cur.copy()
            tsfrm_action_pre[1], tsfrm_action_cur[1] = image_bi_pre.size[1] - tsfrm_action_pre[1], image_bi_cur.size[1] - tsfrm_action_cur[1]
            # angle in [0, 2*pi], angle=2*pi-angle 
            tsfrm_action_pre[2] = 2 * math.pi - tsfrm_action_pre[2]
            tsfrm_action_cur[2] = 2 * math.pi - tsfrm_action_cur[2] 
            return {'image_bi_pre': image_bi_pre, 'image_bi_cur': image_bi_cur, 'image_bi_post': image_bi_post, 'resz_action_pre': tsfrm_action_pre, 'resz_action_cur': tsfrm_action_cur}

class ToTensor(object):
    '''convert ndarrays in sample to tensors
    '''
    def __call__(self, sample):
        image_bi_pre, image_bi_cur, image_bi_post, resz_action_pre, resz_action_cur = sample['image_bi_pre'], sample['image_bi_cur'], sample['image_bi_post'], sample['resz_action_pre'], sample['resz_action_cur']
        # to tensor and binarize image
        image_bi_pre = TF.to_tensor(image_bi_pre) > 0.3
        image_bi_cur = TF.to_tensor(image_bi_cur) > 0.3
        image_bi_post = TF.to_tensor(image_bi_post) > 0.3
        return {'image_bi_pre': image_bi_pre.float(), 'image_bi_cur': image_bi_cur.float(), 'image_bi_post': image_bi_post.float(), 'resz_action_pre': torch.tensor(resz_action_pre), 'resz_action_cur': torch.tensor(resz_action_cur)}

class ToTensorMultiPred4(object):
    '''convert ndarrays in sample to tensors
    '''
    def __call__(self, sample):
        image_bi_pre, image_bi_cur, image_bi_post, image_bi_post2, image_bi_post3, resz_action_pre, resz_action_cur, resz_action_post, resz_action_post2 = \
            sample['image_bi_pre'], sample['image_bi_cur'], sample['image_bi_post'], sample['image_bi_post2'], sample['image_bi_post3'], sample['resz_action_pre'], sample['resz_action_cur'], sample['resz_action_post'], sample['resz_action_post2']
        # to tensor and binarize image
        image_bi_pre = TF.to_tensor(image_bi_pre) > 0.3
        image_bi_cur = TF.to_tensor(image_bi_cur) > 0.3
        image_bi_post = TF.to_tensor(image_bi_post) > 0.3
        image_bi_post2 = TF.to_tensor(image_bi_post2) > 0.3
        image_bi_post3 = TF.to_tensor(image_bi_post3) > 0.3
        return {'image_bi_pre': image_bi_pre.float(), 'image_bi_cur': image_bi_cur.float(), 'image_bi_post': image_bi_post.float(), 'image_bi_post2': image_bi_post2.float(), 'image_bi_post3': image_bi_post3.float(), \
            'resz_action_pre': torch.tensor(resz_action_pre), 'resz_action_cur': torch.tensor(resz_action_cur), 'resz_action_post': torch.tensor(resz_action_post), 'resz_action_post2': torch.tensor(resz_action_post2)}

class ToTensorMultiPred10(object):
    '''convert ndarrays in sample to tensors
    '''
    def __call__(self, sample):
        image_bi_pre, image_bi_cur, image_bi_post, image_bi_post2, image_bi_post3, image_bi_post4, image_bi_post5, image_bi_post6, image_bi_post7, image_bi_post8, image_bi_post9, \
        resz_action_pre, resz_action_cur, resz_action_post, resz_action_post2, resz_action_post3, resz_action_post4, resz_action_post5, resz_action_post6, resz_action_post7, resz_action_post8 = \
            sample['image_bi_pre'], sample['image_bi_cur'], sample['image_bi_post'], sample['image_bi_post2'], sample['image_bi_post3'], sample['image_bi_post4'], sample['image_bi_post5'],\
            sample['image_bi_post6'], sample['image_bi_post7'], sample['image_bi_post8'], sample['image_bi_post9'], \
            sample['resz_action_pre'], sample['resz_action_cur'], sample['resz_action_post'], sample['resz_action_post2'], sample['resz_action_post3'], sample['resz_action_post4'], sample['resz_action_post5'],\
            sample['resz_action_post6'], sample['resz_action_post7'], sample['resz_action_post8']    
        # to tensor and binarize image
        value = 0.1
        image_bi_pre = TF.to_tensor(image_bi_pre) > value
        image_bi_cur = TF.to_tensor(image_bi_cur) > value
        image_bi_post = TF.to_tensor(image_bi_post) > value
        image_bi_post2 = TF.to_tensor(image_bi_post2) > value
        image_bi_post3 = TF.to_tensor(image_bi_post3) > value
        image_bi_post4 = TF.to_tensor(image_bi_post4) > value
        image_bi_post5 = TF.to_tensor(image_bi_post5) > value
        image_bi_post6 = TF.to_tensor(image_bi_post6) > value
        image_bi_post7 = TF.to_tensor(image_bi_post7) > value
        image_bi_post8 = TF.to_tensor(image_bi_post8) > value
        image_bi_post9 = TF.to_tensor(image_bi_post9) > value
        return {'image_bi_pre': image_bi_pre.float(), 'image_bi_cur': image_bi_cur.float(), 'image_bi_post': image_bi_post.float(), 'image_bi_post2': image_bi_post2.float(), 'image_bi_post3': image_bi_post3.float(), \
            'image_bi_post4': image_bi_post4.float(), 'image_bi_post5': image_bi_post5.float(), 'image_bi_post6': image_bi_post6.float(), 'image_bi_post7': image_bi_post7.float(), 'image_bi_post8': image_bi_post8.float(), \
            'image_bi_post9': image_bi_post9.float(), 'resz_action_pre': torch.tensor(resz_action_pre), 'resz_action_cur': torch.tensor(resz_action_cur), 'resz_action_post': torch.tensor(resz_action_post), \
            'resz_action_post2': torch.tensor(resz_action_post2), 'resz_action_post3': torch.tensor(resz_action_post3), 'resz_action_post4': torch.tensor(resz_action_post4), 'resz_action_post5': torch.tensor(resz_action_post5), \
            'resz_action_post6': torch.tensor(resz_action_post6), 'resz_action_post7': torch.tensor(resz_action_post7), 'resz_action_post8': torch.tensor(resz_action_post8)}

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
