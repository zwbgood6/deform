import torch
from torch import nn, optim, sigmoid, tanh, relu
from torch.nn import functional as F
from torchvision.utils import save_image
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import math

class CAE(nn.Module):
    def __init__(self, latent_state_dim=80, latent_act_dim=80):
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
        # action
        self.fc5 = nn.Linear(4, latent_act_dim)
        self.fc6 = nn.Linear(latent_act_dim, latent_act_dim) 
        self.fc7 = nn.Linear(latent_act_dim, latent_act_dim) # 10-100
        self.fc8 = nn.Linear(latent_act_dim, 4)  
        # add these in order to use GPU for parameters
        self.mul_tensor = torch.tensor([50, 50, 2*math.pi, 0.14])
        self.add_tensor = torch.tensor([0, 0, 0, 0.01])
        # latent dim
        #self.latent_act_dim = latent_act_dim
        #self.latent_state_dim = latent_state_dim

    def encoder(self, x):
        if x is None:
            return None
        else:            
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
        #K_T, L_T = self.encoder_matrix(x_cur, a) 
        # print('K_T shape', K_T.shape) 
        # print('L_T shape', L_T.shape)        
        return g_cur, a, g_post, self.decoder(g_cur), self.decoder_act(a)#, K_T, L_T#self.control_matrix

class SysDynamics(nn.Module):
    def __init__(self, latent_state_dim=80, latent_act_dim=80):
        super(SysDynamics, self).__init__()
        self.conv_layers_matrix = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),  
                                            nn.ReLU(),
                                            nn.MaxPool2d(3, stride=1),
                                            nn.Conv2d(32, 64, 3, padding=1),  
                                            nn.ReLU(),
                                            nn.MaxPool2d(3, stride=1),                                            
                                            nn.Conv2d(64, 128, 3, padding=1), 
                                            nn.ReLU(),
                                            nn.MaxPool2d(3, stride=2),
                                            nn.Conv2d(128, 256, 3, padding=1), 
                                            nn.ReLU(),
                                            nn.MaxPool2d(3, stride=2),
                                            nn.Conv2d(256, 512, 3, padding=1), 
                                            nn.ReLU(),
                                            nn.MaxPool2d(3, stride=2),                                            
                                            nn.Conv2d(512, 512, 3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(512, 512, 3, padding=1),
                                            nn.ReLU(),                                            
                                            nn.MaxPool2d(3, stride=2, padding=1)) 
        self.fc31 = nn.Linear(512*2*2, latent_state_dim*latent_state_dim) # K: 9216 -> 6400
        self.fc32 = nn.Linear(latent_state_dim*latent_state_dim, latent_state_dim*latent_state_dim) # K: 9216 -> 6400
        #self.fc32 = nn.Linear(3000, latent_state_dim*latent_state_dim) 
        self.fc41 = nn.Linear(512*2*2 + latent_act_dim, latent_state_dim*latent_act_dim) # L: 9216+40 -> 3200  
        self.fc42 = nn.Linear(latent_state_dim*latent_act_dim, latent_state_dim*latent_act_dim) # L: 9216+40 -> 3200       
        self.fc9 = nn.Linear(4, latent_act_dim)
        self.fc10 = nn.Linear(latent_act_dim, latent_act_dim)
        # latent dim
        self.latent_act_dim = latent_act_dim
        self.latent_state_dim = latent_state_dim

    def encoder_matrix(self, x, a):
        # print('x_cur shape', x_cur.shape)
        # print('x_post shape', x_post.shape)
        #x = torch.cat((x_cur, x_post), 1)
        # print('after concatenation shape', x.shape)
        x = self.conv_layers_matrix(x) # output size: 256*6*6
        # print('after convolution shape', x.shape)
        x = x.view(x.shape[0], -1)
        #print('x shape', x.shape)
        xa = torch.cat((x,a), 1)
        #print('xu shape', xa.shape)
        return relu(self.fc32(relu(self.fc31(x)))).view(-1, self.latent_state_dim, self.latent_state_dim), \
            relu(self.fc42(relu(self.fc41(xa)))).view(-1, self.latent_act_dim, self.latent_state_dim)

    def forward(self, x_cur, u):
        a = relu(self.fc10(relu(self.fc9(u))))  
        K_T, L_T = self.encoder_matrix(x_cur, a) 
        # print('K_T shape', K_T.shape) 
        # print('L_T shape', L_T.shape)        
        return K_T, L_T#self.control_matrix