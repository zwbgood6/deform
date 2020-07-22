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
import math

class CAE(nn.Module):
    def __init__(self, latent_state_dim=80, latent_act_dim=40):
        super(CAE, self).__init__()
        # state
        self.conv_layers = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),  
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2),
                                         nn.Conv2d(32, 64, 3, padding=1), 
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2),
                                         nn.Conv2d(64, 128, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2),
                                         nn.Conv2d(128, 128, 3, padding=1), # channel 1 32 64 64; the next batch size should be larger than 8, 4 corner features + 4 direction features
                                         nn.ReLU(),
                                         nn.Conv2d(128, 128, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2, padding=1))  
        self.fc1 = nn.Linear(128*3*3, latent_state_dim) # size: 128*3*3 > latent_state_dim
        self.fc2 = nn.Linear(latent_state_dim, 128*3*3)       
        self.dconv_layers = nn.Sequential(nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1),
                                          nn.ReLU(),
                                          nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1),
                                          nn.ReLU(), 
                                          nn.ConvTranspose2d(128, 64, 3, stride=2, padding=2),
                                          nn.ReLU(),                                           
                                          nn.ConvTranspose2d(64, 32, 3, stride=2, padding=2),
                                          nn.ReLU(),                                         
                                          nn.ConvTranspose2d(32, 1, 2, stride=2, padding=2),
                                          nn.Sigmoid())
        # K(g^t,g^{t+1}) and L(g^t,g^{t+1},a^t)
        self.conv_layers_matrix = nn.Sequential(nn.Conv2d(2, 64, 3, padding=1),  
                                            nn.ReLU(),
                                            nn.MaxPool2d(3, stride=1),
                                            nn.Conv2d(64, 128, 3, padding=1), 
                                            nn.ReLU(),
                                            nn.MaxPool2d(3, stride=2),
                                            nn.Conv2d(128, 256, 3, padding=1), # channel 1 32 64 64; the next batch size should be larger than 8, 4 corner features + 4 direction features
                                            nn.ReLU(),
                                            nn.MaxPool2d(3, stride=2),
                                            nn.Conv2d(256, 256, 3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(256, 256, 3, padding=1),
                                            nn.ReLU(),                                            
                                            nn.MaxPool2d(3, stride=2, padding=1)) 
        self.fc3 = nn.Linear(256*6*6, latent_state_dim*latent_state_dim) # K: 9216 -> 6400
        #self.fc32 = nn.Linear(3000, latent_state_dim*latent_state_dim) 
        self.fc4 = nn.Linear(256*6*6 + latent_act_dim, latent_state_dim*latent_act_dim) # L: 9216+40 -> 3200
        #self.fc42 = nn.Linear(3200, latent_state_dim*latent_act_dim)    
        # action
        self.fc5 = nn.Linear(4, latent_act_dim)
        self.fc6 = nn.Linear(latent_act_dim, latent_act_dim) 
        self.fc7 = nn.Linear(latent_act_dim, latent_act_dim) # 10-100
        self.fc8 = nn.Linear(latent_act_dim, 4)  
                                                          
        #self.control_matrix = nn.Parameter(torch.rand((latent_state_dim, latent_act_dim), requires_grad=True)) 
        # multiplication/additive to action
        # add these in order to use GPU for parameters
        self.mul_tensor = torch.tensor([50, 50, 2*math.pi, 0.14]) 
        self.add_tensor = torch.tensor([0, 0, 0, 0.01]) 
        # latent dim
        self.latent_act_dim = latent_act_dim
        self.latent_state_dim = latent_state_dim

    def encoder(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1) 
        return relu(self.fc1(x))

    def decoder(self, x):
        x = relu(self.fc2(x))
        x = x.view(-1, 128, 3, 3) #(batch size, channel, H, W)
        return self.dconv_layers(x)
    
    def encoder_act(self, u):
        h1 = relu(self.fc5(u))
        return relu(self.fc6(h1))

    def decoder_act(self, u):
        h2 = relu(self.fc7(u))
        return torch.mul(sigmoid(self.fc8(h2)), self.mul_tensor.cuda()) + self.add_tensor.cuda() 

    def encoder_matrix(self, x_cur, a, x_post):
        # print('x_cur shape', x_cur.shape)
        # print('x_post shape', x_post.shape)
        x = torch.cat((x_cur, x_post), 1)
        # print('after concatenation shape', x.shape)
        x = self.conv_layers_matrix(x) # 256*6*6
        # print('after convolution shape', x.shape)
        x = x.view(x.shape[0], -1)
        #print('x shape', x.shape)
        xa = torch.cat((x,a), 1)
        #print('xu shape', xa.shape)
        return relu(self.fc3(x)).view(-1, self.latent_state_dim, self.latent_state_dim), relu(self.fc4(xa)).view(-1, self.latent_act_dim, self.latent_state_dim)

    def add_identity(self, K):
        # add identity matrix to matrix K
        batch_num, x_len, _ = K.size()       
        return K + torch.eye(x_len).reshape((1, x_len, x_len)).repeat(batch_num, 1, 1).to(device)

    def forward(self, x_cur, u, x_post):
        # print('x_cur shape', x_cur.shape)
        g_cur = self.encoder(x_cur) 
        # print('g_cur shape', g_cur.shape)
        # print('u shape', u.shape)
        a = self.encoder_act(u)  
        # print('a shape', a.shape)
        g_post = self.encoder(x_post)     
        # print('x_cur shape', x_cur.shape)
        # print('a shape', a.shape)
        # print('x_post shape', x_post.shape)
        K_T, L_T = self.encoder_matrix(x_cur, a, x_post) 
        # print('K_T shape', K_T.shape) 
        # print('L_T shape', L_T.shape)        
        return g_cur, a, g_post, self.decoder(g_cur), self.decoder_act(a), K_T, L_T#self.control_matrix


# def get_latent_U(U):
#     U_latent = []           
#     for u in U:
#         u = torch.from_numpy(u).to(device).float().view(-1, 5) 
#         u = model.encoder_act(u).detach().cpu().numpy()
#         U_latent.append(u)
#     n = np.shape(U)[0]        
#     d = np.array(U_latent).shape[2] 
#     return np.resize(np.array(U_latent), (n,d))

def predict():
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # order: img_pre -> act_pre -> img_cur -> act_cur -> img_post
            # previous image 
            img_pre = batch_data['image_bi_pre']
            img_pre = img_pre.float().to(device).view(-1, 1, 50, 50)
            # previous action
            act_pre = batch_data['resz_action_pre']
            act_pre = act_pre.float().to(device).view(-1, 4)
            # current image 
            img_cur = batch_data['image_bi_cur']
            img_cur = img_cur.float().to(device).view(-1, 1, 50, 50) 
            # current action
            act_cur = batch_data['resz_action_cur']
            act_cur = act_cur.float().to(device).view(-1, 4)
            # post image
            img_post = batch_data['image_bi_post']
            img_post = img_post.float().to(device).view(-1, 1, 50, 50)               
            # prediction for current image
            latent_img_pre, latent_act_pre, _, _, _, K_T_pre, L_T_pre = model(img_pre, act_pre, img_cur)
            recon_latent_img_cur = get_next_state_linear(latent_img_pre, latent_act_pre, K_T_pre, L_T_pre)
            recon_img_cur = model.decoder(recon_latent_img_cur)
            # prediction for post image
            latent_img_cur, latent_act_cur, _, _, _, K_T_cur, L_T_cur = model(img_cur, act_cur, img_post)
            recon_latent_img_post = get_next_state_linear(latent_img_cur, latent_act_cur, K_T_cur, L_T_cur)
            recon_img_post = model.decoder(recon_latent_img_post)            
            if batch_idx % 10 == 0:
                n = min(batch_data['image_bi_pre'].size(0), 8)
                comparison = torch.cat([batch_data['image_bi_pre'][:n],
                                        batch_data['image_bi_cur'][:n],
                                        recon_img_cur.view(-1, 1, 50, 50).cpu()[:n],
                                        batch_data['image_bi_post'][:n],
                                        recon_img_post.view(-1, 1, 50, 50).cpu()[:n]])
                save_image(comparison.cpu(),
                         './result/{}/prediction_full_step{}/prediction_batch{}.png'.format(folder_name, step, batch_idx), nrow=n)                                         


print('***** Preparing Data *****')
total_img_num = 1000#22515
image_paths_bi = create_image_path('rope_no_loop_all_resize_gray', total_img_num)
action_path = './rope_dataset/rope_no_loop_all_resize_gray/resize_actions.npy'
actions = np.load(action_path)
dataset = MyDataset(image_paths_bi, actions, transform=ToTensor())   
dataloader = DataLoader(dataset, batch_size=64,
                        shuffle=True, num_workers=4, collate_fn=my_collate)                                             
print('***** Finish Preparing Data *****')

folder_name = 'test'
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
# L = np.load('./result/{}/control_matrix.npy'.format(folder_name))
# L = torch.tensor(L)

# prediction
print('***** Start Prediction *****')
step=1
if not os.path.exists('./result/{}/prediction_full_step{}'.format(folder_name, step)):
    os.makedirs('./result/{}/prediction_full_step{}'.format(folder_name, step))
predict()
print('***** Finish Prediction *****')