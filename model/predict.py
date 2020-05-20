from __future__ import print_function
import argparse
import torch
from torch import nn, optim, sigmoid, tanh, relu
from torch.nn import functional as F
from deform.model.create_dataset import *
from deform.model.hidden_dynamics import *
from torchvision.utils import save_image
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


# def get_latent_U(U):
#     U_latent = []           
#     for u in U:
#         u = torch.from_numpy(u).to(device).float().view(-1, 5) 
#         u = model.encoder_act(u).detach().cpu().numpy()
#         U_latent.append(u)
#     n = np.shape(U)[0]        
#     d = np.array(U_latent).shape[2] 
#     return np.resize(np.array(U_latent), (n,d))

def predict(dataset, actions, L, step):
    # original
    model.eval()
    n = dataset.__len__()
    actions = get_latent_U(actions)
    with torch.no_grad():
        for idx in range(n): 
            data = dataset.__getitem__(idx)
            data = data.float().to(device).view(-1, 1, 50, 50)
            embedded_state = model.encoder(data).cpu().numpy()
            action = actions[idx:idx+step][:]
            next_embedded_state = get_next_state(embedded_state, action, L)
            recon_data = model.decoder(torch.from_numpy(next_embedded_state).float().to(device))
            prediction = recon_data.view(1,50,50)
            save_image(prediction.cpu(), './result/{}/prediction_step{}/predict_{}.png'.format(folder_name, step, idx+step))
    
    # cope from test_new
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

folder_name = 'test_new_CAE3'
PATH = './result/{}/checkpoint'.format(folder_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CAE().to(device)
#optimizer = optim.Adam(model.parameters(), lr=1e-3)

# load check point
print('***** Load Checkpoint *****')
checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
#model.eval()#model.train() 

# dataset
print('***** Load Dataset *****')
run_num = 'run05'
total_img_num = 2353
image_paths = create_image_path(run_num, total_img_num)
dataset = MyDataset(image_paths)

# actions
print('***** Load Actions *****')
actions = get_U(run_num, total_img_num)

# control matrix
print('***** Load Control Matrix *****')
#L = np.ones((10,5))
L = np.load('./result/{}/control_matrix.npy'.format(folder_name))

# prediction
print('***** Start Prediction *****')
model.eval()
step=4
if not os.path.exists('./result/{}/prediction_step{}'.format(folder_name, step)):
    os.makedirs('./result/{}/prediction_step{}'.format(folder_name, step))
predict(dataset, actions, L, step=step)
print('***** Finish Prediction *****')