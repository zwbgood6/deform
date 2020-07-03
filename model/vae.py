from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

def balanced_subsample(x, y, subsample_size=10000):
    sub_x = []
    sub_y = []
    element_count = np.zeros(10)
    break_count = np.zeros(10)
    for i in range(40000):
        if element_count[y[i]] < subsample_size/10:
            sub_x.append(x[i])
            sub_y.append(y[i])
            element_count[y[i]] += 1
        else:
            break_count[y[i]] = 1
        if sum(break_count)==10.0:
            break

    return sub_x, sub_y


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# get train data
train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
# sub-sample
train_len = 10000
train_data = train_dataset.train_data.numpy()
train_labels = train_dataset.train_labels.numpy()
sub_train_data, sub_train_labels = balanced_subsample(train_data, train_labels, subsample_size=train_len)
# reshape
sub_train_data = zoom(sub_train_data, (1, 0.5, 0.5))
# binarize
for i in range(10000):
    for j in range(14):
        for k in range(14):
            if sub_train_data[i, j, k] > 128:
                sub_train_data[i, j, k] = 1
            else:
                sub_train_data[i, j, k] = 0

# get test data
test_dataset = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())
# to numpy
test_data = test_dataset.test_data.numpy()
# reshape
test_data = zoom(test_data, (1, 0.5, 0.5))
# binarize
for i in range(np.shape(test_data)[0]):
    for j in range(14):
        for k in range(14):
            if test_data[i, j, k] > 128:
                test_data[i, j, k] = 1
            else:
                test_data[i, j, k] = 0


train_loader = torch.utils.data.DataLoader(sub_train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(196, 128)
        self.fc21 = nn.Linear(128, 8)
        self.fc22 = nn.Linear(128, 8)
        self.fc3 = nn.Linear(8, 128)
        self.fc4 = nn.Linear(128, 196)
        
    def encode(self, x):
        h1 = F.tanh(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.tanh(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 196))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    bce = F.binary_cross_entropy(recon_x, x.view(-1, 196), reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return bce, kld, bce + kld


def train(epoch):
    model.train()
    train_loss = 0
    bce_all = []
    kld_all = []
    for batch_idx, data in enumerate(train_loader):
        data = data.float().to(device).view(-1, 196)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        bce, kld, loss = loss_function(recon_batch, data, mu, logvar)
        bce_all.append(bce)
        kld_all.append(kld)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    return sum(np.array(bce_all))/len(train_loader.dataset), sum(np.array(kld_all))/len(train_loader.dataset)

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.float().to(device).view(-1, 196)
            recon_batch, mu, logvar = model(data)
            _, _, test_loss_each = loss_function(recon_batch, data, mu, logvar)
            test_loss += test_loss_each.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(8, 196)[:n]]).reshape([-1,1,14,14])
                #save_image(comparison.cpu(),
                #         '/home/zwenbo/Documents/course/ESE546/hw4/results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    bce_all, kld_all = [], []
    for epoch in range(1, args.epochs + 1):
        bce_loss, kld_loss = train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 8).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 14, 14),
                       '/home/zwenbo/Documents/course/ESE546/hw4/results/sample_' + str(epoch) + '.png')
        bce_all.append(-bce_loss)
        kld_all.append(kld_loss)

    x = np.arange(100)

    plt.subplot(2,1,1)
    plt.plot(x, bce_all)
    plt.title('BCE vs epoch')
    plt.ylabel('BCE')
    plt.xlabel('epoch')

    plt.subplot(2,1,2)
    plt.plot(x, kld_all)
    plt.title('KLD vs epoch')
    plt.ylabel('KLD')
    plt.xlabel('epoch')

    #plt.show()    
    plt.savefig('/home/zwenbo/Documents/course/ESE546/hw4/bce_kld.png')