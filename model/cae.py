import torch.nn as nn
import torch.nn.functional as F

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=10):
        super(ConvAutoencoder, self).__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(2, stride=2),
                                         nn.Conv2d(16, 4, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(2, stride=2))
        self.fc1 = nn.Linear(16*3*3, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 16*3*3)
        self.dconv_layers = nn.Sequential(nn.ConvTranspose2d(4, 16, 2, padding=2),
                                          nn.ReLU(),
                                          nn.ConvTranspose2d(16, 1, 2, padding=2),
                                          nn.Sigmoid())
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def encoder(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        return self.fc1(x)

    def decoder(self, x):
        x = self.fc2(x)
        x = x.view(x.shape[0], -1)
        return self.dconv_layers(x)

    def forward(self, x):
        x = self.encoder(x)                
        return self.decoder(x)

# initialize the NN
model = ConvAutoencoder()
print(model)