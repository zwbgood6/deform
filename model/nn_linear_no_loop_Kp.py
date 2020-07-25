from __future__ import print_function
import argparse

import torch
from torch import nn, optim, sigmoid, tanh, relu
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from deform.model.create_dataset import *
from deform.model.hidden_dynamics import *
import matplotlib.pyplot as plt
from deform.utils.utils import plot_train_loss, plot_train_latent_loss, plot_train_img_loss, plot_train_act_loss, plot_train_pred_loss, \
                               plot_test_loss, plot_test_latent_loss, plot_test_img_loss, plot_test_act_loss, plot_test_pred_loss, \
                               plot_sample, rect, save_data, create_loss_list, create_folder
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
        self.conv_layers_matrix = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1),  
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
        #self.fc4 = nn.Linear(256*6*6 + latent_act_dim, latent_state_dim*latent_act_dim) # L: 9216+40 -> 3200
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

    def encoder_matrix(self, x):
        # print('x_cur shape', x_cur.shape)
        # print('x_post shape', x_post.shape)
        #x = torch.cat((x_cur, x_post), 1)
        # print('after concatenation shape', x.shape)
        x = self.conv_layers_matrix(x) # output size: 256*6*6
        # print('after convolution shape', x.shape)
        x = x.view(x.shape[0], -1)
        #print('x shape', x.shape)
        #print('xu shape', xa.shape)
        return relu(self.fc3(x)).view(-1, self.latent_state_dim, self.latent_state_dim)

    def add_identity(self, K):
        # add identity matrix to matrix K
        batch_num, x_len, _ = K.size()       
        return K + torch.eye(x_len).reshape((1, x_len, x_len)).repeat(batch_num, 1, 1).to(device)

    def forward(self, x_cur, x_post):
        # print('x_cur shape', x_cur.shape)
        g_cur = self.encoder(x_cur) 
        # print('g_cur shape', g_cur.shape)
        # print('u shape', u.shape)
        #a = self.encoder_act(u)  
        # print('a shape', a.shape)
        g_post = self.encoder(x_post)     
        # print('x_cur shape', x_cur.shape)
        # print('a shape', a.shape)
        # print('x_post shape', x_post.shape)
        K_T = self.encoder_matrix(x_cur) 
        # print('K_T shape', K_T.shape) 
        # print('L_T shape', L_T.shape)        
        return g_cur, g_post, self.decoder(g_cur), K_T

def loss_function(recon_x, x):
    '''
    recon_x: tensor
    x: tensor
    '''
    return F.binary_cross_entropy(recon_x.view(-1, 2500), x.view(-1, 2500), reduction='sum')
    # mse = torch.nn.BCELoss() # TODO: mseLoss -> binary CELoss
    # return mse(recon_x, x.view(-1, 2500))

def mse(recon_x, x):
    '''mean square error
    recon_x: numpy array
    x: numpy array
    '''
    return F.mse_loss(recon_x, x) 

def loss_function_img(recon_img, img):
    return F.binary_cross_entropy(recon_img.view(-1, 2500), img.view(-1, 2500), reduction='sum')
    # return F.mse_loss(recon_img.view(-1, 2500), img.view(-1, 2500), reduction='sum') # MSE Loss doesnt work

def loss_function_act(recon_act, act):
    # recon_act = torch.div(recon_act.view(-1, 5)[:,:4], torch.tensor([50, 50, 2*math.pi, 0.14])) + torch.tensor([0,0,0,-1/14])
    # act = torch.div(act.view(-1, 5)[:,:4], torch.tensor([50, 50, 2*math.pi, 0.14])) + torch.tensor([0,0,0,-1/14])
    # return F.mse_loss(recon_act, act, reduction='sum')
    # with norm
    return F.mse_loss(recon_act.view(-1, 4), act.view(-1, 4), reduction='sum')
    # return F.mse_loss((torch.div(recon_act.view(-1, 4), torch.tensor([50, 50, 2*math.pi, 0.14, 1])) + torch.tensor([0,0,0,-1/14,0])), 
    #                   (torch.div(act.view(-1, 5), torch.tensor([50, 50, 2*math.pi, 0.14, 1])) + torch.tensor([0,0,0,-1/14,0])), 
    #                   reduction='sum')
    # without norm
    #return F.mse_loss(recon_act.view(-1, 5), act.view(-1, 5), reduction='sum')

# def loss_function_latent(image_pre, image_post, action):
#     G = get_G(image_pre, image_post)
#     U = get_U(action)
#     L = get_control_matrix(G, U)
#     loss_latent = get_error(G, U, L)
#     return loss_latent

def loss_function_latent_linear(latent_img_pre, latent_img_post, latent_action, K_T, L_T):
    # print('latent image pre', latent_img_pre.view(latent_img_pre.shape[0], 1, -1).shape)
    # print('K_T', K_T.shape)
    # print('multiplication', torch.matmul(latent_img_pre.view(latent_img_pre.shape[0], 1, -1), K_T).shape)
    G = latent_img_post.view(latent_img_post.shape[0], 1, -1) - torch.matmul(latent_img_pre.view(latent_img_pre.shape[0], 1, -1), K_T)
    return get_error_linear(G, latent_action, L_T)

def loss_function_latent_linear_without_L(latent_img_pre, latent_img_post, K_T):
    # print('latent image pre', latent_img_pre.view(latent_img_pre.shape[0], 1, -1).shape)
    # print('K_T', K_T.shape)
    # print('multiplication', torch.matmul(latent_img_pre.view(latent_img_pre.shape[0], 1, -1), K_T).shape)
    G = latent_img_post.view(latent_img_post.shape[0], 1, -1) - torch.matmul(latent_img_pre.view(latent_img_pre.shape[0], 1, -1), K_T)
    return torch.norm(G.view(G.shape[0], -1))
    #return get_error_linear(G, latent_action, L_T)

def loss_function_pred_linear(img_post, latent_img_pre, latent_act, K_T, L_T):
    recon_latent_img_post = get_next_state_linear(latent_img_pre, latent_act, K_T, L_T)
    recon_img_post = model.decoder(recon_latent_img_post) 
    return F.binary_cross_entropy(recon_img_post.view(-1, 2500), img_post.view(-1, 2500), reduction='sum')
    #return F.mse_loss(recon_img_post.view(-1, 2500), img_post.view(-1, 2500), reduction='sum') # MSE loss doesn't work

def loss_function_pred_linear_without_L(img_post, latent_img_pre, K_T):
    recon_latent_img_post = get_next_state_linear_without_L(latent_img_pre, K_T)
    recon_img_post = model.decoder(recon_latent_img_post) 
    return F.binary_cross_entropy(recon_img_post.view(-1, 2500), img_post.view(-1, 2500), reduction='sum')

# def get_U(action):         
#     action = action.to(device).float().view(-1, 5) 
#     action = model.encoder_act(action).detach().cpu().numpy() # TODO: change it to tensor rather than array
#     return np.array(action)

# def get_G(image_pre, image_post):
#     image_pre = image_pre.float().to(device).view(-1, 1, 50, 50) 
#     image_post = image_post.float().to(device).view(-1, 1, 50, 50)
#     n = image_pre.shape[0] 
#     latent_pre = model.encoder(image_pre).detach().cpu().numpy().reshape(n,-1)
#     latent_post = model.encoder(image_pre).detach().cpu().numpy().reshape(n,-1)  # TODO: change it to tensor rather than array
#     return latent_post - latent_pre

def constraint_loss(steps, idx, trainset, U_latent, L):
    loss = 0
    data = trainset.__getitem__(idx).float().to(device).view(-1, 1, 50, 50)
    embed_state = model.encoder(data).detach().cpu().numpy()
    for i in range(steps):
        step = i + 1  
        data_next = trainset.__getitem__(idx+step).float().to(device).view(-1, 1, 50, 50)
        action = U_latent[idx:idx+step][:]
        embed_state_next = torch.from_numpy(get_next_state(embed_state, action, L)).to(device).float()
        recon_state_next = model.decoder(embed_state_next)#.detach().cpu()#.numpy()
        loss += mse(recon_state_next, data_next)
    return loss

def train_new(epoch):
    model.train()
    train_loss = 0
    img_loss = 0
    act_loss = 0
    latent_loss = 0
    pred_loss = 0
    for batch_idx, batch_data in enumerate(trainloader):
        # current image before action
        img_cur = batch_data['image_bi_cur']
        img_cur = img_cur.float().to(device).view(-1, 1, 50, 50)
        # action
        # act = batch_data['resz_action_cur']
        # act = act.float().to(device).view(-1, 4)
        # image after action
        img_post = batch_data['image_bi_post']
        img_post = img_post.float().to(device).view(-1, 1, 50, 50)        
        # optimization
        optimizer.zero_grad()
        # model
        latent_img_cur, latent_img_post, recon_img_cur, K_T = model(img_cur, img_post)
        # prediction
        pred_latent_img_post = get_next_state_linear_without_L(latent_img_cur, K_T)
        pred_img_cur = model.decoder(pred_latent_img_post)        
        # loss
        loss_img = loss_function_img(recon_img_cur, img_cur)
        #loss_act = loss_function_act(recon_act, act)
        loss_latent = loss_function_latent_linear_without_L(latent_img_cur, latent_img_post, K_T) # TODO: add prediction loss, decode the latent state to predicted state     
        loss_predict = loss_function_pred_linear_without_L(img_post, latent_img_cur, K_T)
        loss = loss_img + GAMMA_latent * loss_latent + GAMMA_pred * loss_predict
        loss.backward()
        train_loss += loss.item()
        img_loss += loss_img.item()
        #act_loss += GAMMA_act * loss_act.item()
        latent_loss += GAMMA_latent * loss_latent.item()
        pred_loss += GAMMA_pred * loss_predict.item()
        #torch.cuda.empty_cache() # prevent CUDA out of memory RuntimeError
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(batch_data['image_bi_cur']), len(trainloader.dataset),
                100. * batch_idx / len(trainloader),
                loss.item() / len(batch_data['image_bi_cur'])))    
        # reconstruction
        if batch_idx == 0:
            n = min(batch_data['image_bi_cur'].size(0), 8)
            comparison = torch.cat([batch_data['image_bi_cur'][:n],                 # current image
                                  recon_img_cur.view(-1, 1, 50, 50).cpu()[:n],      # reconstruction of current image
                                  batch_data['image_bi_post'][:n],                  # post image
                                  pred_img_cur.view(-1, 1, 50, 50).cpu()[:n]])      # prediction of post image
            save_image(comparison.cpu(),
                     './result/{}/reconstruction_train/reconstruct_epoch_{}.png'.format(folder_name, epoch), nrow=n)      
            # plot_sample(batch_data['image_bi_cur'][:n].detach().cpu().numpy(), 
            #             batch_data['image_bi_post'][:n].detach().cpu().numpy(), 
            #             batch_data['resz_action_cur'][:n].detach().cpu().numpy(), 
            #             recon_act.view(-1, 4)[:n].detach().cpu().numpy(), 
            #             './result/{}/reconstruction_act_train/recon_epoch_{}.png'.format(folder_name, epoch))  
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(trainloader.dataset)))
    n = len(trainloader.dataset)      
    return train_loss/n, img_loss/n, act_loss/n, latent_loss/n, pred_loss/n

def test_new(epoch):
    model.eval()
    test_loss = 0
    img_loss = 0
    act_loss = 0
    latent_loss = 0
    pred_loss = 0    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(testloader):
            # current image before current action
            img_cur = batch_data['image_bi_cur']
            img_cur = img_cur.float().to(device).view(-1, 1, 50, 50)
            # current action
            # act = batch_data['resz_action_cur']
            # act = act.float().to(device).view(-1, 4)
            # post image after current action
            img_post = batch_data['image_bi_post']
            img_post = img_post.float().to(device).view(-1, 1, 50, 50)               
            # model
            latent_img_cur, latent_img_post, recon_img_cur, K_T = model(img_cur, img_post)
            # prediction
            pred_latent_img_post = get_next_state_linear_without_L(latent_img_cur, K_T)
            pred_img_cur = model.decoder(pred_latent_img_post)
            # loss
            loss_img = loss_function_img(recon_img_cur, img_cur)
            #loss_act = loss_function_act(recon_act, act)
            loss_latent = loss_function_latent_linear_without_L(latent_img_cur, latent_img_post, K_T)
            loss_predict = loss_function_pred_linear_without_L(img_post, latent_img_cur, K_T)
            loss = loss_img + GAMMA_latent * loss_latent + GAMMA_pred * loss_predict
            test_loss += loss.item()
            img_loss += loss_img.item()
            #act_loss += GAMMA_act * loss_act.item()
            latent_loss += GAMMA_latent * loss_latent.item()
            pred_loss += GAMMA_pred * loss_predict.item()            
            if batch_idx == 0:
                n = min(batch_data['image_bi_cur'].size(0), 8)
                comparison = torch.cat([batch_data['image_bi_cur'][:n],                 # current image
                                      recon_img_cur.view(-1, 1, 50, 50).cpu()[:n],      # reconstruction of current image
                                      batch_data['image_bi_post'][:n],                  # post image
                                      pred_img_cur.view(-1, 1, 50, 50).cpu()[:n]])      # prediction of post image
                save_image(comparison.cpu(),
                         './result/{}/reconstruction_test/reconstruct_epoch_{}.png'.format(folder_name, epoch), nrow=n)                                         
                # plot_sample(batch_data['image_bi_cur'][:n].detach().cpu().numpy(), 
                #             batch_data['image_bi_post'][:n].detach().cpu().numpy(), 
                #             batch_data['resz_action_cur'][:n].detach().cpu().numpy(), 
                #             recon_act.view(-1, 4)[:n].detach().cpu().numpy(), 
                #             './result/{}/reconstruction_act_test/recon_epoch_{}.png'.format(folder_name, epoch))                           
    n = len(testloader.dataset)
    return test_loss/n, img_loss/n, act_loss/n, latent_loss/n, pred_loss/n

# args
parser = argparse.ArgumentParser(description='CAE Rope Deform Example')
parser.add_argument('--folder-name', default='test', 
                    help='set folder name to save image files')#folder_name = 'test_new_train_scale_large'
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=8, metavar='N',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--gamma-act', type=int, default=450, metavar='N',
                    help='scale coefficient for loss of action (default: 150*3)')   
parser.add_argument('--gamma-lat', type=int, default=900, metavar='N',
                    help='scale coefficient for loss of latent dynamics (default: 150*6)')     
parser.add_argument('--gamma-pred', type=int, default=5, metavar='N',
                    help='scale coefficient for loss of prediction (default: 3)')                                                          
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--math', default=False,
                    help='get control matrix L: True - use regression, False - use backpropagation')                    
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--restore', action='store_true', default=False)                    
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)


# dataset
print('***** Preparing Data *****')
total_img_num = 1000#22515
train_num = int(total_img_num * 0.8)
image_paths_bi = create_image_path('rope_no_loop_all_resize_gray', total_img_num)
#image_paths_ori = create_image_path('rope_all_ori', total_img_num)
resz_act_path = './rope_dataset/rope_no_loop_all_resize_gray/resize_actions.npy'
#ori_act_path = './rope_dataset/rope_all_ori/actions.npy'
resz_act = np.load(resz_act_path)
#ori_act = np.load(ori_act_path)
# transform = transforms.Compose([Translation(10), 
#                                 HFlip(0.5), 
#                                 VFlip(0.5), 
#                                 ToTensor()])   
transform = transforms.Compose([Translation(10), 
                                ToTensor()])                           
#dataset = MyDataset(image_paths_bi, resz_act, transform=ToTensor())
trainset = MyDataset(image_paths_bi[0:train_num], resz_act[0:train_num], transform=transform)
testset = MyDataset(image_paths_bi[train_num:], resz_act[train_num:], transform=ToTensor())
trainloader = DataLoader(trainset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, collate_fn=my_collate)
testloader = DataLoader(testset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, collate_fn=my_collate)                        
print('***** Finish Preparing Data *****')

# train var
MATH = args.math # True: use regression; False: use backpropagation
GAMMA_act = args.gamma_act
GAMMA_latent = args.gamma_lat
GAMMA_pred = args.gamma_pred

# create folders
folder_name = args.folder_name
create_folder(folder_name)

print('***** Start Training & Testing *****')
device = torch.device("cuda" if args.cuda else "cpu")
epochs = args.epochs
model = CAE().to(device)
#torch.cuda.empty_cache()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
#optimizer = optim.SGD(model.parameters(), lr=1e-3)

# initial train
if not args.restore:
    init_epoch = 1
    loss_logger = None
# restore previous train        
else:
    print('***** Load Checkpoint *****')
    PATH = './result/{}/checkpoint'.format(folder_name)
    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])  
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    init_epoch = checkpoint['epoch'] + 1
    loss_logger = checkpoint['loss_logger'] 

train_loss_all, train_img_loss_all, train_act_loss_all, train_latent_loss_all, train_pred_loss_all, _, \
test_loss_all, test_img_loss_all, test_act_loss_all, test_latent_loss_all, test_pred_loss_all, _ = create_loss_list(loss_logger, kld=False)         

# freeze the layers for K and L
for param in model.conv_layers_matrix.parameters():
    param.requires_grad = False
for param in model.fc3.parameters():
    param.requires_grad = False    
# for param in model.fc4.parameters():
#     param.requires_grad = False


for epoch in range(init_epoch, epochs+1):
    # freeze the layers for g^t and a^t
    # unfreeze the layers for K and L
    if epoch == 4: # hyper-param      
        # for param in model.conv_layers.parameters():
        #     param.requires_grad = False
        # for param in model.fc1.parameters():
        #     param.requires_grad = False
        # for param in model.fc2.parameters():
        #     param.requires_grad = False 
        # for param in model.dconv_layers.parameters():
        #     param.requires_grad = False              
        for param in model.conv_layers_matrix.parameters():
            param.requires_grad = True            
        for param in model.fc3.parameters():
            param.requires_grad = True          
        # for param in model.fc4.parameters():
        #     param.requires_grad = True                           
        # for param in model.fc5.parameters():
        #     param.requires_grad = False 
        # for param in model.fc6.parameters():
        #     param.requires_grad = False 
        # for param in model.fc7.parameters():
        #     param.requires_grad = False 
        # for param in model.fc8.parameters():
        #     param.requires_grad = False                                                         
    train_loss, train_img_loss, train_act_loss, train_latent_loss, train_pred_loss = train_new(epoch)
    test_loss, test_img_loss, test_act_loss, test_latent_loss, test_pred_loss = test_new(epoch)
    train_loss_all.append(train_loss)
    train_img_loss_all.append(train_img_loss)
    train_act_loss_all.append(train_act_loss)
    train_latent_loss_all.append(train_latent_loss)
    train_pred_loss_all.append(train_pred_loss)
    test_loss_all.append(test_loss)
    test_img_loss_all.append(test_img_loss)
    test_act_loss_all.append(test_act_loss)
    test_latent_loss_all.append(test_latent_loss)
    test_pred_loss_all.append(test_pred_loss)    
    if epoch % args.log_interval == 0:
        save_data(folder_name, epochs, train_loss_all, train_img_loss_all, train_act_loss_all,
                  train_latent_loss_all, train_pred_loss_all, test_loss_all, test_img_loss_all,
                  test_act_loss_all, test_latent_loss_all, test_pred_loss_all, None, None, None, None)        
        # save checkpoint
        PATH = './result/{}/checkpoint'.format(folder_name)
        loss_logger = {'train_loss_all': train_loss_all, 'train_img_loss_all': train_img_loss_all, 
                       'train_act_loss_all': train_act_loss_all, 'train_latent_loss_all': train_latent_loss_all,
                       'train_pred_loss_all': train_pred_loss_all, 'test_loss_all': test_loss_all,
                       'test_img_loss_all': test_img_loss_all, 'test_act_loss_all': test_act_loss_all, 
                       'test_latent_loss_all': test_latent_loss_all, 'test_pred_loss_all': test_pred_loss_all}
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_logger': loss_logger
                    }, 
                    PATH)

save_data(folder_name, epochs, train_loss_all, train_img_loss_all, train_act_loss_all,
          train_latent_loss_all, train_pred_loss_all, test_loss_all, test_img_loss_all,
          test_act_loss_all, test_latent_loss_all, test_pred_loss_all, None, None, None, None)

# plot
plot_train_loss('./result/{}/train_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_train_img_loss('./result/{}/train_img_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_train_act_loss('./result/{}/train_act_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_train_latent_loss('./result/{}/train_latent_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_train_pred_loss('./result/{}/train_pred_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_test_loss('./result/{}/test_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_test_img_loss('./result/{}/test_img_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_test_act_loss('./result/{}/test_act_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_test_latent_loss('./result/{}/test_latent_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_test_pred_loss('./result/{}/test_pred_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)

# save checkpoint
PATH = './result/{}/checkpoint'.format(folder_name)
loss_logger = {'train_loss_all': train_loss_all, 'train_img_loss_all': train_img_loss_all, 
               'train_act_loss_all': train_act_loss_all, 'train_latent_loss_all': train_latent_loss_all,
               'train_pred_loss_all': train_pred_loss_all, 'test_loss_all': test_loss_all,
               'test_img_loss_all': test_img_loss_all, 'test_act_loss_all': test_act_loss_all, 
               'test_latent_loss_all': test_latent_loss_all, 'test_pred_loss_all': test_pred_loss_all}
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_logger': loss_logger
            }, 
            PATH)
print('***** End Program *****')            

