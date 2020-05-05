from __future__ import print_function
import argparse

import torch
from torch import nn, optim, sigmoid, tanh, relu
from torch.nn import functional as F
from torchvision.utils import save_image
from deform.model.create_dataset import *
from deform.model.hidden_dynamics import *
import matplotlib.pyplot as plt
from deform.utils.utils import plot_train_loss, plot_error
import os

class AE(nn.Module):    
    def __init__(self):
        super(AE, self).__init__()
        self.fc1 = nn.Linear(5, 100)
        self.fc2 = nn.Linear(100, 1000) # TODO: add ConV, max pooling, and add layers
        self.fc3 = nn.Linear(1000, 100) # 10-100
        self.fc4 = nn.Linear(100, 5)  

    def encoder(self, x):
        h1 = relu(self.fc1(x)) # relu -> tanh for all relu's # TODO: relu
        return relu(self.fc2(h1))

    def decoder(self, g):
        h2 = relu(self.fc3(g))
        return sigmoid(self.fc4(h2))   

    def forward(self, x):
        x = self.encoder(x.view(-1, 5))
        return self.decoder(x)  

class CAE(nn.Module):
    def __init__(self, latent_state_dim=500, latent_act_dim=100):
        super(CAE, self).__init__()
        # state
        self.conv_layers = nn.Sequential(nn.Conv2d(1, 8, 7, padding=0), # kernel size 3
                                         nn.ReLU(),
                                         nn.MaxPool2d(7, stride=2),
                                         nn.Conv2d(8, 16, 7, padding=0), # 64 channels, increase number of layers
                                         nn.ReLU(),
                                         nn.MaxPool2d(7, stride=2))  # TODO: four five layer
        self.fc1 = nn.Linear(16*4*4, latent_state_dim)
        self.fc2 = nn.Linear(latent_state_dim, 16*4*4)
        self.dconv_layers = nn.Sequential(nn.ConvTranspose2d(16, 8, 7, stride=3, padding=0),
                                          nn.ReLU(),
                                          nn.ConvTranspose2d(8, 1, 5, stride=3, padding=0),
                                          nn.Sigmoid())
        # action
        self.fc5 = nn.Linear(5, 30)
        self.fc6 = nn.Linear(30, latent_act_dim) # TODO: add ConV, max pooling, and add layers
        self.fc7 = nn.Linear(latent_act_dim, 30) # 10-100
        self.fc8 = nn.Linear(30, 5)  

    def encoder(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)    
        return relu(self.fc1(x))

    def decoder(self, x):
        x = relu(self.fc2(x))
        x = x.view(-1, 16, 4, 4) #(batch size, channel, H, W)
        return self.dconv_layers(x)
    
    def encoder_act(self, u):
        h1 = relu(self.fc5(u)) # relu -> tanh for all relu's # TODO: relu
        return relu(self.fc6(h1))

    def decoder_act(self, u):
        h2 = relu(self.fc7(u))
        return sigmoid(self.fc8(h2))   

    def forward(self, x, u):
        x = self.encoder(x) 
        u = self.encoder_act(u)               
        return self.decoder(x), self.decoder_act(u)


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

def get_latent_U(U):
    U_latent = []           
    for u in U:
        u = torch.from_numpy(u).to(device).float().view(-1, 5) 
        u = model.encoder_act(u).detach().cpu().numpy()
        U_latent.append(u)
    n = np.shape(U)[0]        
    d = np.array(U_latent).shape[2] 
    return np.resize(np.array(U_latent), (n,d))

def train(model, trainset, epochs, step):   
    n = trainset.__len__()
    error = []
    train_loss_all = []
    err_tmp = np.inf
    U = get_U(run_num, train_num)
    L = None
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for idx in range(n): 
            # optimization
            optimizer.zero_grad()
            # state
            data = trainset.__getitem__(idx)
            data = data.float().to(device).view(-1, 1, 50, 50)            
            # action
            action = torch.from_numpy(U[idx]).to(device).float().view(-1, 5)     
            #action = model.encoder_act(action).view(-1, 5)     
            # model
            recon_data, recon_act = model(data, action)
            # loss 
            loss = GAMMA1 * loss_function(recon_data, data) 
            loss_act = GAMMA2 * F.mse_loss(recon_act, action) 
            loss += loss_act
            # # loss of 1->k steps
            # if L is not None and idx > 0 and idx < n-step:
            #     model.eval()
            #     U_latent = get_latent_U(U)
            #     loss += GAMMA1 * constraint_loss(step, idx, trainset, U_latent, L)
            #     model.train()

            # loss of all steps
            if idx == n-1:
                model.eval()
                G = get_G(model, trainset)
                #U = get_U(run_num, train_num)
                U_latent = get_latent_U(U)
                U_latent = U_latent[:-1, :]
                L = get_control_matrix(G, U_latent)
                err_tmp = get_error(G, U_latent, L)
                error.append(err_tmp)
                loss += torch.tensor(err_tmp)#loss_function(recon_data, data) + GAMMA2 * torch.tensor(err_tmp)
                model.train() 
            loss.backward()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
            train_loss += loss.item()
            optimizer.step()
        # get final control matrix L    
        if epoch == epochs-1:
            np.save('./result/{}/control_matrix.npy'.format(folder_name), L)  # TODO: unit test    
        train_loss = train_loss / n    
        train_loss_all.append(train_loss)  

        # # get final control matrix L  
        # if epoch == epochs-1:
        #     model.eval()
        #     G = get_G(model, trainset)
        #     #U = get_U(run_num, train_num)
        #     L = get_control_matrix(G, U)
        #     np.save('./result/{}/control_matrix.npy'.format(folder_name), L) 
        print("epoch : {}/{}, loss = {:.6f}, error = {:.10f}".format(epoch + 1, epochs, train_loss, err_tmp))
        # if err_tmp < 1:
        #     break
        
        #recon_data = model(data)

    return train_loss_all, error

def test(dataset):
    model.eval()
    test_loss = 0
    n = dataset.__len__()
    with torch.no_grad():
        for idx in range(n): 
            data = dataset.__getitem__(idx)
            data = data.float().to(device).view(-1, 1, 50, 50)
            recon_data = model.decoder(model.encoder(data))
            loss = loss_function(recon_data, data)
            test_loss += loss.item()
            comparison = recon_data.view(50,50)
            save_image(comparison.cpu(), './result/{}/reconstruction/reconstruct_'.format(folder_name) + str(idx) + '.png')
    test_loss /= n
    print('Test set loss: {:.4f}'.format(test_loss))
    return test_loss

# dataset
print('***** Preparing Data *****')
run_num='run05'
total_img_num = 2353
train_num = int(total_img_num * 0.8)
image_paths = create_image_path(run_num, total_img_num)
dataset = MyDataset(image_paths)
trainset = MyDataset(image_paths[0:train_num])
testset = MyDataset(image_paths[train_num:])
print('***** Finish Preparing Data *****')

# train
print('***** Start Training *****')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 2
GAMMA1 = 10
GAMMA2 = 1
folder_name = 'test_new_CAE3'

if not os.path.exists('./result/' + folder_name):
    os.makedirs('./result/' + folder_name)
if not os.path.exists('./result/' + folder_name + '/plot'):
    os.makedirs('./result/' + folder_name + '/plot')
if not os.path.exists('./result/' + folder_name + '/reconstruction'):
    os.makedirs('./result/' + folder_name + '/reconstruction')

train_loss, error = train(model, trainset, epochs, step=1)
np.save('./result/{}/train_loss_epoch{}.npy'.format(folder_name, epochs), train_loss)
np.save('./result/{}/error_epoch{}.npy'.format(folder_name, epochs), error)

# test
print('***** Start Testing *****')
test_loss = test(dataset) # TODO: get reconstruction for trainset
np.save('./result/{}/test_loss_epoch{}.npy'.format(folder_name, epochs), test_loss)

# plot
plot_train_loss('./result/{}/train_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_error('./result/{}/error_epoch{}.npy'.format(folder_name, epochs), folder_name)

print('***** End Program *****')

# save checkpoint
PATH = './result/{}/checkpoint'.format(folder_name)
torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss
            }, 
            PATH)

