from __future__ import print_function
import argparse
import torch
from torch import nn, optim, sigmoid, tanh, relu
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from deform.model.create_dataset import *
from deform.model.hidden_dynamics import *
from torchvision.utils import save_image
import os

class CAE(nn.Module):
    def __init__(self, latent_state_dim=500, latent_act_dim=100):
        super(CAE, self).__init__()
        # state
        self.conv_layers = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),  
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2),
                                         nn.Conv2d(32, 64, 3, padding=1), 
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2),
                                         nn.Conv2d(64, 64, 3, padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(64, 64, 3, padding=1), # channel 1 32 64 64; the next batch size should be larger than 8, 4 corner features + 4 direction features
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2))  # TODO: add conv relu conv relu max
        self.fc1 = nn.Linear(64*5*5, latent_state_dim) # TODO: 64*2*2 > latent_state_dim
        self.fc2 = nn.Linear(latent_state_dim, 64*5*5)
        self.dconv_layers = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, stride=2, padding=2),
                                          nn.ReLU(),
                                          nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1),
                                          nn.ReLU(), 
                                          nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),
                                          nn.ReLU(),                                         
                                          nn.ConvTranspose2d(32, 1, 2, stride=2, padding=0),
                                          nn.Sigmoid())
        # action
        self.fc5 = nn.Linear(5, 30)
        self.fc6 = nn.Linear(30, latent_act_dim) 
        self.fc7 = nn.Linear(latent_act_dim, 30) # 10-100
        self.fc8 = nn.Linear(30, 5)  
        # control matrix
        self.control_matrix = nn.Parameter(torch.rand((latent_state_dim, latent_act_dim), requires_grad=True)) # TODO: okay for random initializaion?

    def encoder(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1) 
        return relu(self.fc1(x))

    def decoder(self, x):
        x = relu(self.fc2(x))
        x = x.view(-1, 64, 5, 5) #(batch size, channel, H, W)
        return self.dconv_layers(x)
    
    def encoder_act(self, u):
        h1 = relu(self.fc5(u))
        return relu(self.fc6(h1))

    def decoder_act(self, u):
        h2 = relu(self.fc7(u))
        return sigmoid(self.fc8(h2))   

    def forward(self, x_pre, u, x_post):
        x_pre = self.encoder(x_pre) 
        u = self.encoder_act(u)  
        x_post = self.encoder(x_post)     
        return x_pre, u, x_post, self.decoder(x_pre), self.decoder_act(u), self.control_matrix

# def get_latent_U(U):
#     U_latent = []           
#     for u in U:
#         u = torch.from_numpy(u).to(device).float().view(-1, 5) 
#         u = model.encoder_act(u).detach().cpu().numpy()
#         U_latent.append(u)
#     n = np.shape(U)[0]        
#     d = np.array(U_latent).shape[2] 
#     return np.resize(np.array(U_latent), (n,d))

def predict(L):
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # image before action
            img_pre = batch_data['image_pre']
            img_pre = img_pre.float().to(device).view(-1, 1, 50, 50)
            # action
            act = batch_data['action']
            act = act.float().to(device).view(-1, 5)
            # image after action
            img_post = batch_data['image_post']
            img_post = img_post.float().to(device).view(-1, 1, 50, 50)               
            # model
            latent_img_pre, latent_act, _, _, _, _ = model(img_pre, act, img_post)
            recon_latent_img_post = get_next_state(latent_img_pre, latent_act, L)
            recon_img_post = model.decoder(recon_latent_img_post)
            if batch_idx % 10 == 0:
                n = min(batch_data['image_pre'].size(0), 8)
                comparison = torch.cat([batch_data['image_post'][:n],
                                      recon_img_post.view(-1, 1, 50, 50).cpu()[:n]])
                save_image(comparison.cpu(),
                         './result/{}/prediction_step{}/prediction_batch{}.png'.format(folder_name, step, batch_idx), nrow=n)                                         


print('***** Preparing Data *****')
total_img_num = 1000#77944
image_paths = create_image_path(total_img_num)
action_path = './rope_dataset/rope_all_resize_gray/resize_actions.npy'
actions = np.load(action_path)
dataset = MyDataset(image_paths, actions)   
dataloader = DataLoader(dataset, batch_size=64,
                        shuffle=True, num_workers=4)                                             
print('***** Finish Preparing Data *****')

folder_name = 'test_new_train1'
PATH = './result/{}/checkpoint'.format(folder_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CAE().to(device)

# load check point
print('***** Load Checkpoint *****')
checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
#model.eval()#model.train() 

# # dataset
# print('***** Load Dataset *****')
# run_num = 'run05'
# total_img_num = 2353
# image_paths = create_image_path(run_num, total_img_num)
# dataset = MyDataset(image_paths)

# # actions
# print('***** Load Actions *****')
# actions = get_U(run_num, total_img_num)

# control matrix
print('***** Load Control Matrix *****')
#L = np.ones((10,5))
L = np.load('./result/{}/control_matrix.npy'.format(folder_name))
L = torch.tensor(L)

# prediction
print('***** Start Prediction *****')
step=1
if not os.path.exists('./result/{}/prediction_step{}'.format(folder_name, step)):
    os.makedirs('./result/{}/prediction_step{}'.format(folder_name, step))
predict(L)
print('***** Finish Prediction *****')