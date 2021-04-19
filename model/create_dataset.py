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
    def __init__(self, image_paths_bi, resize_actions, transform=None):
        self.image_paths_bi = image_paths_bi 
        self.resz_actions = resize_actions
        self.transform = transform

    def __getitem__(self, index):  
        n = self.__len__()
        none_sample = {'image_bi_pre': None, 'image_bi_cur': None, 'image_bi_post': None, 'resz_action_pre': None, 'resz_action_cur': None}
        if index == 0:
            if n == 2:
                index = 1
            else:    
                index = np.random.randint(1, n-1)
        if index == n-1:
            if n == 2:
                index = 1
            else:    
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
        
        sample = {'image_bi_pre': image_bi_pre, 'image_bi_cur': image_bi_cur, 'image_bi_post': image_bi_post, 'resz_action_pre': resz_action_pre[:4], 'resz_action_cur': resz_action_cur[:4]}       
 
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_paths_bi)

class MyDatasetMultiPred4(Dataset):  
    def __init__(self, image_paths_bi, resize_actions, transform=None):
        self.image_paths_bi = image_paths_bi 
        self.resz_actions = resize_actions
        self.transform = transform

    def __getitem__(self, index):  
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

        sample = {'image_bi_pre': image_bi_pre, 'image_bi_cur': image_bi_cur, 'image_bi_post': image_bi_post, 'image_bi_post2': image_bi_post2, 'image_bi_post3': image_bi_post3,\
            'resz_action_pre': resz_action_pre[:4], 'resz_action_cur': resz_action_cur[:4], 'resz_action_post': resz_action_post[:4], 'resz_action_post2': resz_action_post2[:4]}       

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_paths_bi)

class MyDatasetMultiPred10(Dataset):    
    def __init__(self, image_paths_bi, resize_actions, transform=None):
        self.image_paths_bi = image_paths_bi 
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

   
        sample = {'image_bi_pre': image_bi_pre, 'image_bi_cur': image_bi_cur, 'image_bi_post': image_bi_post, 'image_bi_post2': image_bi_post2, 'image_bi_post3': image_bi_post3,\
            'image_bi_post4': image_bi_post4, 'image_bi_post5': image_bi_post5, 'image_bi_post6': image_bi_post6, 'image_bi_post7': image_bi_post7, 'image_bi_post8': image_bi_post8, \
            'image_bi_post9': image_bi_post9, \
            'resz_action_pre': resz_action_pre[:4], 'resz_action_cur': resz_action_cur[:4], 'resz_action_post': resz_action_post[:4], 'resz_action_post2': resz_action_post2[:4], \
            'resz_action_post3': resz_action_post3[:4], 'resz_action_post4': resz_action_post4[:4], 'resz_action_post5': resz_action_post5[:4], 'resz_action_post6': resz_action_post6[:4], \
            'resz_action_post7': resz_action_post7[:4], 'resz_action_post8': resz_action_post8[:4]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_paths_bi)

class Translation(object):
    '''Translate the image and action [-max_translation, max_translation]. e.g., [-10, 10]
    '''
    def __init__(self, max_translation=10):
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


class HFlip(object):
    '''Ramdom Horizontal flip the image and action with probabilty p
    '''
    def __init__(self, probability=0.5):
        self.p = probability

    def flip_angle(self, action):
        if action[2] > math.pi:
            action[2] = 3 * math.pi - action[2] 
        else:
            action[2] = math.pi - action[2] 
        return action

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        else:    
            image_bi_pre, image_bi_cur, image_bi_post, action_pre, action_cur = sample['image_bi_pre'], sample['image_bi_cur'], sample['image_bi_post'], sample['resz_action_pre'], sample['resz_action_cur']
            image_bi_pre, image_bi_cur, image_bi_post = TF.hflip(image_bi_pre), TF.hflip(image_bi_cur), TF.hflip(image_bi_post)

            tsfrm_action_pre, tsfrm_action_cur = action_pre.copy(), action_cur.copy()
            tsfrm_action_pre[0], tsfrm_action_cur[0] = image_bi_pre.size[0] - tsfrm_action_pre[0], image_bi_cur.size[0] - tsfrm_action_cur[0]

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

            tsfrm_action_pre, tsfrm_action_cur = action_pre.copy(), action_cur.copy()
            tsfrm_action_pre[1], tsfrm_action_cur[1] = image_bi_pre.size[1] - tsfrm_action_pre[1], image_bi_cur.size[1] - tsfrm_action_cur[1]

            tsfrm_action_pre[2] = 2 * math.pi - tsfrm_action_pre[2]
            tsfrm_action_cur[2] = 2 * math.pi - tsfrm_action_cur[2] 
            return {'image_bi_pre': image_bi_pre, 'image_bi_cur': image_bi_cur, 'image_bi_post': image_bi_post, 'resz_action_pre': tsfrm_action_pre, 'resz_action_cur': tsfrm_action_cur}

class ToTensor(object):
    '''convert ndarrays in sample to tensors
    '''
    def __call__(self, sample):
        image_bi_pre, image_bi_cur, image_bi_post, resz_action_pre, resz_action_cur = sample['image_bi_pre'], sample['image_bi_cur'], sample['image_bi_post'], sample['resz_action_pre'], sample['resz_action_cur']
        image_bi_pre = TF.to_tensor(image_bi_pre) > 0.3
        image_bi_cur = TF.to_tensor(image_bi_cur) > 0.3
        image_bi_post = TF.to_tensor(image_bi_post) > 0.3              
        return {'image_bi_pre': image_bi_pre.float(), 'image_bi_cur': image_bi_cur.float(), 'image_bi_post': image_bi_post.float(), 'resz_action_pre': torch.tensor(resz_action_pre), 'resz_action_cur': torch.tensor(resz_action_cur)}

class ToTensorRGB(object):
    '''convert ndarrays in sample to tensors
    '''
    def __call__(self, sample):
        image_bi_pre, image_bi_cur, image_bi_post, resz_action_pre, resz_action_cur = sample['image_bi_pre'], sample['image_bi_cur'], sample['image_bi_post'], sample['resz_action_pre'], sample['resz_action_cur']
        image_bi_pre = TF.to_tensor(image_bi_pre) 
        image_bi_cur = TF.to_tensor(image_bi_cur) 
        image_bi_post = TF.to_tensor(image_bi_post)               
        return {'image_bi_pre': image_bi_pre.float(), 'image_bi_cur': image_bi_cur.float(), 'image_bi_post': image_bi_post.float(), 'resz_action_pre': torch.tensor(resz_action_pre), 'resz_action_cur': torch.tensor(resz_action_cur)}


class ToTensorMultiPred4(object):
    '''convert ndarrays in sample to tensors
    '''
    def __call__(self, sample):
        image_bi_pre, image_bi_cur, image_bi_post, image_bi_post2, image_bi_post3, resz_action_pre, resz_action_cur, resz_action_post, resz_action_post2 = \
            sample['image_bi_pre'], sample['image_bi_cur'], sample['image_bi_post'], sample['image_bi_post2'], sample['image_bi_post3'], sample['resz_action_pre'], sample['resz_action_cur'], sample['resz_action_post'], sample['resz_action_post2']
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

