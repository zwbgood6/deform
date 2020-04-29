from __future__ import print_function
import argparse
import torch
from torch import nn, optim, sigmoid, tanh, relu
from torch.nn import functional as F
from deform.model.create_dataset import *
from deform.model.hidden_dynamics import *
from torchvision.utils import save_image
import os

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

class CAE(nn.Module):
    def __init__(self, latent_dim=10):
        super(CAE, self).__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(2, stride=2),
                                         nn.Conv2d(16, 4, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(2, stride=2))
        self.fc1 = nn.Linear(4*12*12, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 4*12*12)
        self.dconv_layers = nn.Sequential(nn.ConvTranspose2d(4, 16, 5, stride=2, padding=1),
                                          nn.ReLU(),
                                          nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
                                          nn.Sigmoid())

    def encoder(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        return self.fc1(x)

    def decoder(self, x):
        x = self.fc2(x)
        x = x.view(-1, 4, 12, 12) #(batch size, channel, H, W)
        return self.dconv_layers(x)

    def forward(self, x):
        x = self.encoder(x)                
        return self.decoder(x)

def predict(dataset, actions, L, step=1):
    n = dataset.__len__()
    with torch.no_grad():
        for idx in range(n): 
            data = dataset.__getitem__(idx)
            data = data.float().to(device).view(-1, 1, 50, 50)
            embedded_state = model.encoder(data).cpu().numpy()
            action = actions[idx:idx+step][:]
            next_embedded_state = get_next_state(embedded_state, action, L)
            recon_data = model.decoder(torch.from_numpy(next_embedded_state).float().to(device))
            prediction = recon_data.view(50,50)
            save_image(prediction.cpu(), './result/{}/prediction_step{}/predict_{}.png'.format(folder_name, step, idx+step))


folder_name = 'test_CAE'
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
step=1
if not os.path.exists('./result/{}/prediction_step{}'.format(folder_name, step)):
    os.makedirs('./result/{}/prediction_step{}'.format(folder_name, step))
predict(dataset, actions, L, step)
print('***** Finish Prediction *****')