import torch
from torch import nn, optim, sigmoid, tanh, relu
from torch.nn import functional as F

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


folder_name = 'test_new_train_math_F'
PATH = './result/{}/checkpoint'.format(folder_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CAE().to(device)
#optimizer = optim.Adam(model.parameters(), lr=1e-3)

# load check point
print('***** Load Checkpoint *****')
checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
print('test')