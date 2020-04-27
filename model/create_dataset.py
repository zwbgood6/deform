'''
learn from https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/5
'''

import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, image_paths, train=True):
        self.image_paths = image_paths

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        x = self.transform(image)
        # binarize image
        x = x > 0.5 
        x = x.float()
        return x

    def __len__(self):
        return len(self.image_paths)

    def transform(self, image):
        image = TF.to_grayscale(image)
        resize = transforms.Resize(size=(50, 50))
        image = resize(image)        
        image = TF.to_tensor(image) 
        return image
                    

def create_image_path(run_num, total_img_num):
    '''create image_path list as input of MyDataset
    run_num: string, e.g., 'run03'
    total_img_num: number of images under 'run03'
    '''
    add1 = './rope_dataset/rope_seg/'
    add2 = run_num
    image_paths = []
    for i in range(total_img_num):
        if len(str(i)) == 1:
            add3 = '/img_000{}.jpg'.format(i)
        elif len(str(i)) == 2:
            add3 = '/img_00{}.jpg'.format(i)
        elif len(str(i)) == 3:
            add3 = '/img_0{}.jpg'.format(i)   
        elif len(str(i)) == 4:
            add3 = '/img_{}.jpg'.format(i)                   
        image_paths.append(add1+add2+add3)
    return image_paths

# #pass unit test
# if __name__ == "__main__":
#     image_paths = create_image_path(run_num='run05', total_img_num=100)
#     dataset = MyDataset(image_paths)
#     n = dataset.__len__()
#     for i in range(n):
#         x = dataset.__getitem__(i)
#         x_ = x.view(-1, 128)
