'''
learn from https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/5
'''

import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, image_paths, actions):
        self.image_paths = image_paths
        self.actions = actions

    def __getitem__(self, index):
        n = self.__len__()
        if index == n-1:
            index = index - 1
        # transform images
        image_pre = Image.open(self.image_paths[index])
        image_post = Image.open(self.image_paths[index+1])
        current_size = np.array(image_pre).shape[0]
        output_size = 50
        image_pre = self.transform_img(image_pre, output_size)
        image_post = self.transform_img(image_post, output_size)
        # transform x, y positions in action
        action = self.actions[index]
        ratio = output_size / current_size
        action = self.transform_act(action, ratio)
        # sample = {state, action, next_state}
        sample = {'image_pre': image_pre, 'action': action, 'image_post': image_post}
        return sample

    def __len__(self):
        return len(self.image_paths)

    def transform_img(self, image, output_size):
        image = TF.to_grayscale(image)
        resize = transforms.Resize(size=(output_size, output_size))
        image = resize(image)        
        image = TF.to_tensor(image) 
        # binarize image
        image = image > 0.5 
        image = image.float()
        return image

    def transform_act(self, action, ratio):
        action[:2] = action[:2] * ratio
        return action


def create_image_path(total_img_num):
    '''create image_path list as input of MyDataset
    total_img_num: number of images 
    '''
    add1 = './rope_dataset/rope_all'
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
