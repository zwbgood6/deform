from __future__ import print_function
import argparse

import torch
from torch import nn, optim, sigmoid, tanh, relu
from torch.nn import functional as F
from torchvision.utils import save_image
from model.create_dataset import *
from model.hidden_dynamics import *
import matplotlib.pyplot as plt
from utils import plot_train_loss, plot_error

class AE(nn.Module):    
    def __init__(self):
        super(AE, self).__init__()
        self.fc1 = nn.Linear(2500, 250)
        self.fc2 = nn.Linear(250, 10)
        self.fc3 = nn.Linear(10, 250)
        self.fc4 = nn.Linear(250, 2500)  

    def encoder(self, x):
        h1 = tanh(self.fc1(x)) # relu -> tanh for all relu's
        return tanh(self.fc2(h1))

    def decoder(self, g):
        h2 = tanh(self.fc3(g))
        return sigmoid(self.fc4(h2))   

    def forward(self, x):
        x = self.encoder(x.view(-1, 2500))
        return self.decoder(x)  


def loss_function(recon_x, x):
    return F.binary_cross_entropy(recon_x, x.view(-1, 2500), reduction='sum')
    # mse = torch.nn.BCELoss() # TODO: mseLoss -> binary CELoss
    # return mse(recon_x, x.view(-1, 2500))


def train(model, trainset, epochs):   
    n = trainset.__len__()
    error = []
    train_loss_all = []
    err_tmp = np.inf
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for idx in range(n): 
            data = trainset.__getitem__(idx)
            data = data.float().to(device).view(-1, 50*50)
            optimizer.zero_grad()
            recon_data = model(data)
            loss = loss_function(recon_data, data)
            loss.backward() 
            train_loss += loss.item()
            optimizer.step()
        train_loss = train_loss / n    
        train_loss_all.append(train_loss)    
        if epoch % 1 == 0:
            model.eval()
            G = get_G(model, trainset)
            U = get_U(run_num, train_num)
            L = get_control_matrix(G, U)
            err_tmp = get_error(G, U, L)
            error.append(err_tmp)
            print("epoch : {}/{}, loss = {:.6f}, error = {:.10f}".format(epoch + 1, epochs, train_loss, err_tmp))
        # if err_tmp < 1:
        #     break
        recon_data = model(data)

    return train_loss_all, error

def test(testset):
    model.eval()
    test_loss = 0
    n = testset.__len__()
    with torch.no_grad():
        for idx in range(n): 
            data = testset.__getitem__(idx)
            data = data.float().to(device).view(-1, 50*50)
            recon_data = model(data)
            loss = loss_function(recon_data, data)
            test_loss += loss.item()
            comparison = recon_data.view(50,50)
            save_image(comparison.cpu(), './result/reconstruction/reconstruct_' + str(idx) + '.png')
    test_loss /= n
    print('Test set loss: {:.4f}'.format(test_loss))
    return test_loss

# dataset
print('***** Preparing Data *****')
run_num='run05'
total_img_num = 2353
train_num = int(total_img_num * 0.8)
image_paths = create_image_path(run_num, total_img_num)
trainset = MyDataset(image_paths[0:train_num])
testset = MyDataset(image_paths[train_num:])
print('***** Finish Preparing Data *****')

# train
print('***** Start Training *****')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 100

train_loss, error = train(model, trainset, epochs)
np.save('./result/train_loss_100.npy', train_loss)
np.save('./result/error_100.npy', error)

# test
print('***** Start Testing *****')
test_loss = test(testset)
np.save('./result/test_loss_100.npy', test_loss)

# plot
plot_train_loss('./result/train_loss_100.npy')
plot_error('./result/error_100.npy')

print('***** End Program *****')

# save checkpoint
PATH = './checkpoint'
torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss
            }, 
            PATH)

# # load check point
# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# model.train() # or model.eval()