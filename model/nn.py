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
        self.fc1 = nn.Linear(2500, 250)
        self.fc2 = nn.Linear(250, 10) # TODO: add ConV, max pooling, and add layers
        self.fc3 = nn.Linear(10, 250) # 10-100
        self.fc4 = nn.Linear(250, 2500)  

    def encoder(self, x):
        h1 = tanh(self.fc1(x)) # relu -> tanh for all relu's # TODO: relu
        return tanh(self.fc2(h1))

    def decoder(self, g):
        h2 = tanh(self.fc3(g))
        return sigmoid(self.fc4(h2))   

    def forward(self, x):
        x = self.encoder(x.view(-1, 2500))
        return self.decoder(x)  

class CAE(nn.Module):
    def __init__(self, latent_dim=16):
        super(CAE, self).__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(2, stride=2),
                                         nn.Conv2d(8, 16, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(2, stride=2))
        self.fc1 = nn.Linear(16*12*12, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 16*12*12)
        self.dconv_layers = nn.Sequential(nn.ConvTranspose2d(16, 8, 3, stride=2, padding=0),
                                          nn.ReLU(),
                                          nn.ConvTranspose2d(8, 1, 2, stride=2, padding=0),
                                          nn.Sigmoid())

    def encoder(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        return self.fc1(x)

    def decoder(self, x):
        x = self.fc2(x)
        x = x.view(-1, 16, 12, 12) #(batch size, channel, H, W)
        return self.dconv_layers(x)

    def forward(self, x):
        x = self.encoder(x)                
        return self.decoder(x)


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

def constraint_loss(steps, idx, trainset, U, L):
    loss = 0
    data = trainset.__getitem__(idx).float().to(device).view(-1, 1, 50, 50)
    embed_state = model.encoder(data).detach().cpu().numpy()
    for i in range(steps):
        step = i + 1  
        data_next = trainset.__getitem__(idx+step).float().to(device).view(-1, 1, 50, 50)
        action = U[idx:idx+step][:]
        embed_state_next = torch.from_numpy(get_next_state(embed_state, action, L)).float()
        recon_state_next = model.decoder(embed_state_next).detach().cpu()#.numpy()
        loss += mse(recon_state_next, data_next)
    return loss

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
            data = trainset.__getitem__(idx)
            data = data.float().to(device).view(-1, 1, 50, 50)
            #latent = model.encoder(data).detach().cpu().numpy().reshape(-1).tolist()
            optimizer.zero_grad()
            recon_data = model(data)
            # loss of autoencoder
            loss = loss_function(recon_data, data)
            # loss of 1->k steps
            if L is not None and idx > 0 and idx < n-step:
                model.eval()
                loss += GAMMA1 * constraint_loss(step, idx, trainset, U, L)
                model.train()
            # loss of all steps
            if idx == n-1:
                model.eval()
                G = get_G(model, trainset)
                #U = get_U(run_num, train_num)
                L = get_control_matrix(G, U)
                err_tmp = get_error(G, U, L)
                error.append(err_tmp)
                loss += GAMMA2 * torch.tensor(err_tmp)#loss_function(recon_data, data) + GAMMA2 * torch.tensor(err_tmp)
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
        recon_data = model(data)

    return train_loss_all, error

def test(dataset):
    model.eval()
    test_loss = 0
    n = dataset.__len__()
    with torch.no_grad():
        for idx in range(n): 
            data = dataset.__getitem__(idx)
            data = data.float().to(device).view(-1, 1, 50, 50)
            recon_data = model(data)
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
#model = AE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 1
GAMMA1 = 100
GAMMA2 = 1
folder_name = 'test_CAE'

if not os.path.exists('./result/' + folder_name):
    os.makedirs('./result/' + folder_name)
if not os.path.exists('./result/' + folder_name + '/plot'):
    os.makedirs('./result/' + folder_name + '/plot')
if not os.path.exists('./result/' + folder_name + '/reconstruction'):
    os.makedirs('./result/' + folder_name + '/reconstruction')

train_loss, error = train(model, trainset, epochs, step=4)
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

