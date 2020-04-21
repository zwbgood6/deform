import torch
from torch import nn, optim, sigmoid, relu
from torchvision.utils import save_image
from model.create_dataset import *
PATH = './checkpoint'

class AE(nn.Module):    
    def __init__(self):
        super(AE, self).__init__()
        self.fc1 = nn.Linear(2500, 250)
        self.fc2 = nn.Linear(250, 10)
        self.fc3 = nn.Linear(10, 250)
        self.fc4 = nn.Linear(250, 2500)  

    def encoder(self, x):
        h1 = relu(self.fc1(x)) # relu -> tanh for all relu's
        return relu(self.fc2(h1))

    def decoder(self, g):
        h2 = relu(self.fc3(g))
        return sigmoid(self.fc4(h2))   

    def forward(self, x):
        x = self.encoder(x.view(-1, 2500))
        return self.decoder(x)  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# load check point
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
#model.train() 
model.eval()

run_num='run04'
idx = 9
total_img_num = 10
train_num = int(total_img_num * 0.8)
image_paths = create_image_path(run_num, total_img_num)
data = MyDataset(image_paths[0:train_num])
data = data.__getitem__(idx)
data = data.float().to(device).view(-1, 50*50)
recon_data = model(data)
comparison = recon_data.view(50, 50)
save_image(comparison.cpu(), './result/reconstruction/reconstruct_' + str(idx) + '.png')