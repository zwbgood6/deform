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
from deform.utils.utils import plot_train_loss, plot_test_loss, plot_latent_loss, plot_img_loss, plot_act_loss
import os

class CAE(nn.Module):
    def __init__(self, latent_state_dim=500, latent_act_dim=100):
        super(CAE, self).__init__()
        # state
        self.conv_layers = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), # 
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2),
                                         nn.Conv2d(8, 16, 3, padding=1), # 
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2),
                                         nn.Conv2d(16, 32, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2),
                                         nn.Conv2d(32, 64, 3, padding=1), # channel 1 32 64 64 
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2))  # TODO: add conv relu conv relu max
        self.fc1 = nn.Linear(64*2*2, latent_state_dim)
        self.fc2 = nn.Linear(latent_state_dim, 64*2*2)
        self.dconv_layers = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, stride=3, padding=1),
                                          nn.ReLU(),
                                          nn.ConvTranspose2d(32, 16, 3, stride=3, padding=1),
                                          nn.ReLU(), 
                                          nn.ConvTranspose2d(16, 8, 3, stride=3, padding=2),
                                          nn.ReLU(),                                         
                                          nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
                                          nn.Sigmoid())
        # action
        self.fc5 = nn.Linear(5, 30)
        self.fc6 = nn.Linear(30, latent_act_dim) # TODO: add ConV, max pooling, and add layers
        self.fc7 = nn.Linear(latent_act_dim, 30) # 10-100
        self.fc8 = nn.Linear(30, 5)  
        # control matrix
        #self.control_matrix = nn.Parameter(torch.tensor(init_value, requires_grad=True)) # TODO: backpropagation this matrix
        self.control_matrix = nn.Parameter(torch.rand((latent_state_dim, latent_act_dim), requires_grad=True))

    def encoder(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1) 
        return relu(self.fc1(x))

    def decoder(self, x):
        x = relu(self.fc2(x))
        x = x.view(-1, 64, 2, 2) #(batch size, channel, H, W)
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
        return x_pre, u, x_post, self.decoder(x_pre), self.decoder_act(u), self.control_matrix # TODO: change it / done


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

def loss_function_act(recon_act, act):
    return F.mse_loss(recon_act.view(-1, 5), act.view(-1, 5))

# def loss_function_latent(image_pre, image_post, action):
#     G = get_G(image_pre, image_post)
#     U = get_U(action)
#     L = get_control_matrix(G, U)
#     loss_latent = get_error(G, U, L)
#     return loss_latent

def loss_function_latent(latent_img_pre, latent_img_post, latent_action, L, math=True):
    G = latent_img_post - latent_img_pre
    if math:
        L = get_control_matrix(G, latent_action)
    return get_error(G, latent_action, L), L

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
    for batch_idx, batch_data in enumerate(trainloader):
        # image before action
        img_pre = batch_data['image_pre']
        img_pre = img_pre.float().to(device).view(-1, 1, 50, 50)
        # action
        act = batch_data['action']
        act = act.float().to(device).view(-1, 5)
        # image after action
        img_post = batch_data['image_post']
        img_post = img_post.float().to(device).view(-1, 1, 50, 50)        
        # optimization
        optimizer.zero_grad()
        # model
        latent_img_pre, latent_act, latent_img_post, recon_img_pre, recon_act, L_bp = model(img_pre, act, img_post)
        # loss
        loss_img = loss_function_img(recon_img_pre, img_pre)
        loss_act = loss_function_act(recon_act, act)
        loss_latent, L = loss_function_latent(latent_img_pre, latent_img_post, latent_act, L_bp, math=MATH)
        loss = loss_img + loss_act + loss_latent
        loss.backward()
        train_loss += loss.item()
        img_loss += loss_img.item()
        act_loss += loss_act.item()
        latent_loss += loss_latent.item()
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(batch_data['image_pre']), len(trainloader.dataset),
                100. * batch_idx / len(trainloader),
                loss.item() / len(batch_data['image_pre'])))    

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(trainloader.dataset)))
    n = len(trainloader.dataset)      
    return train_loss/n, img_loss/n, act_loss/n, latent_loss/n, L

def test_new(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(testloader):
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
            latent_img_pre, latent_act, latent_img_post, recon_img_pre, recon_act, L = model(img_pre, act, img_post)
            # loss
            loss_img = loss_function_img(recon_img_pre, img_pre)
            loss_act = loss_function_act(recon_act, act)
            loss_latent, _ = loss_function_latent(latent_img_pre, latent_img_post, latent_act, L, math=MATH)
            loss = loss_img + loss_act + loss_latent
            test_loss += loss.item()
            if batch_idx == 0:
                n = min(batch_data['image_pre'].size(0), 8)
                comparison = torch.cat([batch_data['image_pre'][:n],
                                      recon_img_pre.view(-1, 1, 50, 50).cpu()[:n]])
                save_image(comparison.cpu(),
                         './result/{}/reconstruction_test/reconstruct_epoch_'.format(folder_name) + str(epoch) + '.png', nrow=n)                                         
        for batch_idx, batch_data in enumerate(trainloader):
            if batch_idx == 0:
                n = min(batch_data['image_pre'].size(0), 8)
                comparison = torch.cat([batch_data['image_pre'][:n],
                                      recon_img_pre.view(-1, 1, 50, 50).cpu()[:n]])
                save_image(comparison.cpu(),
                         './result/{}/reconstruction_train/reconstruct_epoch_'.format(folder_name) + str(epoch) + '.png', nrow=n)             
    n = len(testloader.dataset)
    return test_loss/n

# dataset
print('***** Preparing Data *****')
#run_num='run05'
total_img_num = 1000#77944
train_num = int(total_img_num * 0.8)
image_paths = create_image_path(total_img_num)
# image_path = './rope_dataset/rope_all'
action_path = './rope_dataset/rope_all_resize_gray/resize_actions.npy'
actions = np.load(action_path)
dataset = MyDataset(image_paths, actions)
trainset = MyDataset(image_paths[0:train_num], actions[0:train_num])
testset = MyDataset(image_paths[train_num:], actions[train_num:])
trainloader = DataLoader(trainset, batch_size=64,
                        shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=64,
                        shuffle=True, num_workers=4)                        
print('***** Finish Preparing Data *****')

# train
MATH = True # True: do math; False: do backpropagation
print('***** Start Training & Testing *****')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 5
folder_name = 'test_new_train1'

if not os.path.exists('./result/' + folder_name):
    os.makedirs('./result/' + folder_name)
if not os.path.exists('./result/' + folder_name + '/plot'):
    os.makedirs('./result/' + folder_name + '/plot')
if not os.path.exists('./result/' + folder_name + '/reconstruction_test'):
    os.makedirs('./result/' + folder_name + '/reconstruction_test')
if not os.path.exists('./result/' + folder_name + '/reconstruction_train'):
    os.makedirs('./result/' + folder_name + '/reconstruction_train')

train_loss_all = []
img_loss_all = []
act_loss_all = []
latent_loss_all = []
test_loss_all = []

for epoch in range(1, epochs+1):
    train_loss, img_loss, act_loss, latent_loss, L = train_new(epoch)
    test_loss = test_new(epoch)
    train_loss_all.append(train_loss)
    img_loss_all.append(img_loss)
    act_loss_all.append(act_loss)
    latent_loss_all.append(latent_loss)
    test_loss_all.append(test_loss)
#train_loss, error = train(model, trainset, epochs, step=1)
    if epoch % 10 == 0:
        np.save('./result/{}/train_loss_epoch{}.npy'.format(folder_name, epochs), train_loss_all)
        np.save('./result/{}/img_loss_epoch{}.npy'.format(folder_name, epochs), img_loss_all)
        np.save('./result/{}/act_loss_epoch{}.npy'.format(folder_name, epochs), act_loss_all)
        np.save('./result/{}/latent_loss_epoch{}.npy'.format(folder_name, epochs), latent_loss_all)
        np.save('./result/{}/test_loss_epoch{}.npy'.format(folder_name, epochs), test_loss_all)
        #np.save('./result/{}/control_matrix.npy'.format(folder_name), L.detach().numpy()) # TODO: does it influence the result?
        # save checkpoint
        PATH = './result/{}/checkpoint'.format(folder_name)
        torch.save({
                    'epoch': epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss
                    }, 
                    PATH)

np.save('./result/{}/train_loss_epoch{}.npy'.format(folder_name, epochs), train_loss_all)
np.save('./result/{}/img_loss_epoch{}.npy'.format(folder_name, epochs), img_loss_all)
np.save('./result/{}/act_loss_epoch{}.npy'.format(folder_name, epochs), act_loss_all)
np.save('./result/{}/latent_loss_epoch{}.npy'.format(folder_name, epochs), latent_loss_all)
np.save('./result/{}/test_loss_epoch{}.npy'.format(folder_name, epochs), test_loss_all)
np.save('./result/{}/control_matrix.npy'.format(folder_name), L.detach().numpy())
# test
# print('***** Start Testing *****')
# test_loss = test(dataset) # TODO: get reconstruction for trainset
# np.save('./result/{}/test_loss_epoch{}.npy'.format(folder_name, epochs), test_loss)

# plot
plot_train_loss('./result/{}/train_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_test_loss('./result/{}/test_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_img_loss('./result/{}/img_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_act_loss('./result/{}/act_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_latent_loss('./result/{}/latent_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
#plot_error('./result/{}/error_epoch{}.npy'.format(folder_name, epochs), folder_name)

# save checkpoint
PATH = './result/{}/checkpoint'.format(folder_name)
torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss
            }, 
            PATH)



print('***** End Program *****')            


# class AE(nn.Module):    
#     def __init__(self):
#         super(AE, self).__init__()
#         self.fc1 = nn.Linear(5, 100)
#         self.fc2 = nn.Linear(100, 1000) # TODO: add ConV, max pooling, and add layers
#         self.fc3 = nn.Linear(1000, 100) # 10-100
#         self.fc4 = nn.Linear(100, 5)  

#     def encoder(self, x):
#         h1 = relu(self.fc1(x)) # relu -> tanh for all relu's # TODO: relu
#         return relu(self.fc2(h1))

#     def decoder(self, g):
#         h2 = relu(self.fc3(g))
#         return sigmoid(self.fc4(h2))   

#     def forward(self, x):
#         x = self.encoder(x.view(-1, 5))
#         return self.decoder(x)  

# def train(model, trainset, epochs, step):   
#     n = trainset.__len__()
#     error = []
#     train_loss_all = []
#     err_tmp = np.inf
#     U = get_U(run_num, train_num)
#     L = None
#     for epoch in range(epochs):
#         model.train()
#         train_loss = 0
#         for idx in range(n): 
#             # optimization
#             optimizer.zero_grad()
#             # state
#             data = trainset.__getitem__(idx)
#             data = data.float().to(device).view(-1, 1, 50, 50)            
#             # action
#             action = torch.from_numpy(U[idx]).to(device).float().view(-1, 5)     
#             #action = model.encoder_act(action).view(-1, 5)
#             # model
#             recon_data, recon_act = model(data, action)
#             # loss 
#             loss = GAMMA1 * loss_function(recon_data, data) 
#             loss_act = GAMMA2 * F.mse_loss(recon_act, action) 
#             loss += loss_act
#             # # loss of 1->k steps
#             # if L is not None and idx > 0 and idx < n-step:
#             #     model.eval()
#             #     U_latent = get_latent_U(U)
#             #     loss += GAMMA1 * constraint_loss(step, idx, trainset, U_latent, L)
#             #     model.train()

#             # loss of all steps
#             if idx == n-1:
#                 G = get_G(model, trainset)
#                 #U = get_U(run_num, train_num)
#                 U_latent = get_latent_U(U)
#                 U_latent = U_latent[:-1, :]
#                 L = get_control_matrix(G, U_latent)
#                 err_tmp = get_error(G, U_latent, L)
#                 error.append(err_tmp)
#                 loss += torch.tensor(err_tmp)#loss_function(recon_data, data) + GAMMA2 * torch.tensor(err_tmp)
#             loss.backward()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
#             train_loss += loss.item()
#             optimizer.step()
#         # get final control matrix L    
#         if epoch == epochs-1:
#             np.save('./result/{}/control_matrix.npy'.format(folder_name), L)  # TODO: unit test    
#         train_loss = train_loss / n    
#         train_loss_all.append(train_loss)  

#         # # get final control matrix L  
#         # if epoch == epochs-1:
#         #     model.eval()
#         #     G = get_G(model, trainset)
#         #     #U = get_U(run_num, train_num)
#         #     L = get_control_matrix(G, U)
#         #     np.save('./result/{}/control_matrix.npy'.format(folder_name), L) 
#         print("epoch : {}/{}, loss = {:.6f}, error = {:.10f}".format(epoch + 1, epochs, train_loss, err_tmp))
#         # if err_tmp < 1:
#         #     break
        
#         #recon_data = model(data)
#     return train_loss_all, error

# def test(dataset):
#     model.eval()
#     test_loss = 0
#     n = dataset.__len__()
#     with torch.no_grad():
#         for idx in range(n): 
#             data = dataset.__getitem__(idx)
#             data = data.float().to(device).view(-1, 1, 50, 50)
#             recon_data = model.decoder(model.encoder(data))
#             loss = loss_function(recon_data, data)
#             test_loss += loss.item()
#             comparison = recon_data.view(50,50)
#             save_image(comparison.cpu(), './result/{}/reconstruction/reconstruct_'.format(folder_name) + str(idx) + '.png')
#     test_loss /= n
#     print('Test set loss: {:.4f}'.format(test_loss))
#     return test_loss