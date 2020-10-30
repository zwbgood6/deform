# separate two models, train g^t first, then train K and L
from __future__ import print_function
import argparse

import torch
from torch import nn, optim, sigmoid, tanh, relu
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from deform.model.create_dataset import *
from deform.model.hidden_dynamics import *
import matplotlib.pyplot as plt
from deform.utils.utils import plot_train_loss, plot_train_bound_loss, plot_train_kl_loss, plot_train_pred_loss, \
                               plot_test_loss, plot_test_bound_loss, plot_test_kl_loss, plot_test_pred_loss, \
                               plot_sample, rect, save_data, save_e2c_data, create_loss_list, create_folder, plot_grad_flow
from deform.model.configs import load_config
import os
import math

class NormalDistribution(object):
    """
    Wrapper class representing a multivariate normal distribution parameterized by
    N(mu,Cov). If cov. matrix is diagonal, Cov=(sigma).^2. Otherwise,
    Cov=A*(sigma).^2*A', where A = (I+v*r^T).
    """

    def __init__(self, mu, sigma, logsigma, *, v=None, r=None):
        self.mu = mu
        self.sigma = sigma
        self.logsigma = logsigma
        self.v = v
        self.r = r

    @property
    def cov(self):
        """This should only be called when NormalDistribution represents one sample"""
        if self.v is not None and self.r is not None:
            assert self.v.dim() == 1
            dim = self.v.dim()
            v = self.v.unsqueeze(1)  # D * 1 vector
            rt = self.r.unsqueeze(0)  # 1 * D vector
            A = torch.eye(dim) + v.mm(rt)
            return A.mm(torch.diag(self.sigma.pow(2)).mm(A.t()))
        else:
            return torch.diag(self.sigma.pow(2))


def KLDGaussian(Q, N, eps=1e-8):
    """KL Divergence between two Gaussians
        Assuming Q ~ N(mu0, A\sigma_0A') where A = I + vr^{T}
        and      N ~ N(mu1, \sigma_1)
    """
    sum = lambda x: torch.sum(x, dim=1)
    k = float(Q.mu.size()[1])  # dimension of distribution
    mu0, v, r, mu1 = Q.mu, Q.v, Q.r, N.mu
    s02, s12 = (Q.sigma).pow(2) + eps, (N.sigma).pow(2) + eps
    a = sum(s02 * (1. + 2. * v * r) / s12) + sum(v.pow(2) / s12) * sum(r.pow(2) * s02)  # trace term
    b = sum((mu1 - mu0).pow(2) / s12)  # difference-of-means term
    c = 2. * (sum(N.logsigma - Q.logsigma) - torch.log(1. + sum(v * r) + eps))  # ratio-of-determinants term.

    #
    # print('trace: %s' % a)
    # print('mu_diff: %s' % b)
    # print('k: %s' % k)
    # print('det: %s' % c)

    return 0.5 * (a + b - k + c)

class E2C(nn.Module):
    def __init__(self, dim_z=80, dim_u=4):
        super(E2C, self).__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),  
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2),
                                         nn.Conv2d(32, 64, 3, padding=1), 
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2),
                                         nn.Conv2d(64, 128, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2),
                                         nn.Conv2d(128, 128, 3, padding=1), # channel 1 32 64 64; the next batch size should be larger than 8, 4 corner features + 4 direction features
                                         nn.ReLU(),
                                         nn.Conv2d(128, 128, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2, padding=1))          
        self.fc1 = nn.Linear(128*3*3, dim_z*2)
        self.fc2 = nn.Linear(dim_z, 128*3*3)  
        self.dconv_layers = nn.Sequential(nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1),
                                          nn.ReLU(),
                                          nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1),
                                          nn.ReLU(), 
                                          nn.ConvTranspose2d(128, 64, 3, stride=2, padding=2),
                                          nn.ReLU(),                                           
                                          nn.ConvTranspose2d(64, 32, 3, stride=2, padding=2),
                                          nn.ReLU(),                                         
                                          nn.ConvTranspose2d(32, 1, 2, stride=2, padding=2),
                                          nn.Sigmoid())
        self.trans = nn.Sequential(
                        nn.Linear(dim_z, 100),
                        nn.BatchNorm1d(100),
                        nn.ReLU(),
                        nn.Linear(100, 100),
                        nn.BatchNorm1d(100),
                        nn.ReLU(),
                        nn.Linear(100, dim_z*2)
        )    
        self.fc_B = nn.Linear(dim_z, dim_z * dim_u)
        self.fc_o = nn.Linear(dim_z, dim_z)   
        self.dim_z = dim_z
        self.dim_u = dim_u                                   


    def encode(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1) 
        return relu(self.fc1(x)).chunk(2, dim=1)

    def decode(self, x):
        x = relu(self.fc2(x))
        x = x.view(-1, 128, 3, 3)
        return self.dconv_layers(x)

    def transition(self, h, Q, u):
        batch_size = h.size()[0]
        v, r = self.trans(h).chunk(2, dim=1)
        v1 = v.unsqueeze(2).cpu()
        rT = r.unsqueeze(1).cpu()
        I = Variable(torch.eye(self.dim_z).repeat(batch_size, 1, 1))
        if rT.data.is_cuda:
            I.data.cuda()
        A = I.add(v1.bmm(rT)).cuda()

        B = self.fc_B(h).view(-1, self.dim_z, self.dim_u)
        o = self.fc_o(h).reshape((-1, self.dim_z, 1))

        # need to compute the parameters for distributions
        # as well as for the samples
        u = u.unsqueeze(2)

        d = A.bmm(Q.mu.unsqueeze(2)).add(B.bmm(u)).add(o).squeeze(2)
        sample = A.bmm(h.unsqueeze(2)).add(B.bmm(u)).add(o).squeeze(2)        
        return sample, NormalDistribution(d, Q.sigma, Q.logsigma, v=v, r=r)

    def reparam(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        self.z_mean = mean
        self.z_sigma = std
        eps = torch.FloatTensor(std.size()).normal_()
        #if std.data.is_cuda:
        #    eps.cuda()
        eps = Variable(eps)
        return eps.mul(std.cpu()).add_(mean.cpu()).cuda(), NormalDistribution(mean, std, torch.log(std))

    def forward(self, x, action, x_next):
        mean, logvar = self.encode(x)
        mean_next, logvar_next = self.encode(x_next)

        z, self.Qz = self.reparam(mean, logvar)
        z_next, self.Qz_next = self.reparam(mean_next, logvar_next)

        self.x_dec = self.decode(z)
        self.x_next_dec = self.decode(z_next)

        self.z_next_pred, self.Qz_next_pred = self.transition(z, self.Qz, action)
        self.x_next_pred_dec = self.decode(self.z_next_pred)

        return self.x_dec, self.x_next_pred_dec, self.Qz, self.Qz_next, self.Qz_next_pred

    def latent_embeddings(self, x):
        return self.encode(x)[0]

    def predict(self, X, U):
        mean, logvar = self.encode(X)
        z, Qz = self.reparam(mean, logvar)
        z_next_pred, Qz_next_pred = self.transition(z, Qz, U)
        return self.decode(z_next_pred)

def binary_crossentropy(t, o, eps=1e-8):
    return t * torch.log(o + eps) + (1.0 - t) * torch.log(1.0 - o + eps)

def compute_loss(x_dec, x_next_pred_dec, x, x_next,
                 Qz, Qz_next_pred,
                 Qz_next):
    # Reconstruction losses
    if False:
        x_reconst_loss = (x_dec - x_next).pow(2).sum(dim=1)
        x_next_reconst_loss = (x_next_pred_dec - x_next).pow(2).sum(dim=1)
    else:
        x_reconst_loss = -binary_crossentropy(x, x_dec).sum(dim=1)
        x_next_reconst_loss = -binary_crossentropy(x_next, x_next_pred_dec).sum(dim=1)

    logvar = Qz.logsigma.mul(2)
    KLD_element = Qz.mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element, dim=1).mul(-0.5)

    # ELBO
    bound_loss = x_reconst_loss.add(x_next_reconst_loss).add(KLD.reshape(-1,1,1))
    kl = KLDGaussian(Qz_next_pred, Qz_next)
    kl = kl[~torch.isnan(kl)]
    return bound_loss.mean(), kl.mean()

def train(e2c_model):
    e2c_model.train()
    bound_loss = 0
    kl_loss = 0
    train_loss = 0
    pred_loss = 0
    for batch_idx, batch_data in enumerate(trainloader):
        # current image before action
        x = batch_data['image_bi_cur']
        x = x.float().to(device).view(-1, 1, 50, 50)
        # action
        action = batch_data['resz_action_cur']
        action = action.float().to(device).view(-1, 4)
        # image after action
        x_next = batch_data['image_bi_post']
        x_next = x_next.float().to(device).view(-1, 1, 50, 50)        
        # optimization
        e2c_optimizer.zero_grad() 
        # model
        x_dec, x_next_pred_dec, Qz, Qz_next, Qz_next_pred = e2c_model(x, action, x_next)
        # prediction
        x_next_pred = e2c_model.predict(x, action)
        # loss
        loss_pred = F.binary_cross_entropy(x_next_pred.view(-1, 2500), x_next.view(-1, 2500), reduction='sum')            
        loss_bound, loss_kl = compute_loss(x_dec, x_next_pred_dec, x, x_next, Qz, Qz_next_pred, Qz_next)
        loss = GAMMA_bound * loss_bound + GAMMA_kl * loss_kl + GAMMA_pred * loss_pred        
        loss.backward()
        train_loss += loss.item()
        bound_loss += GAMMA_bound * loss_bound.item()
        kl_loss += GAMMA_kl * loss_kl.item()
        pred_loss += GAMMA_pred * loss_pred.item()
        e2c_optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(batch_data['image_bi_cur']), len(trainloader.dataset),
                100. * batch_idx / len(trainloader),
                loss.item() / len(batch_data['image_bi_cur'])))    
        # reconstruction
        if batch_idx == 0:
            n = min(batch_data['image_bi_cur'].size(0), 8)
            comparison = torch.cat([batch_data['image_bi_cur'][:n],                 # current image
                                x_dec.view(-1, 1, 50, 50).cpu()[:n]])      # reconstruction of current image                                                 
            save_image(comparison.cpu(),
                    './result/{}/reconstruction_train/reconstruct_epoch_{}.png'.format(folder_name, epoch), nrow=n)      
            # plot_sample(batch_data['image_bi_cur'][:n].detach().cpu().numpy(), 
            #             batch_data['image_bi_post'][:n].detach().cpu().numpy(), 
            #             batch_data['resz_action_cur'][:n].detach().cpu().numpy(), 
            #             recon_act.view(-1, 4)[:n].detach().cpu().numpy(), 
            #             './result/{}/reconstruction_act_train/recon_epoch_{}.png'.format(folder_name, epoch))  
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(trainloader.dataset)))
    n = len(trainloader.dataset)      
    return train_loss/n, bound_loss/n, kl_loss/n, pred_loss/n

def test(e2c_model):
    e2c_model.eval()
    test_loss = 0
    bound_loss = 0
    kl_loss = 0
    pred_loss = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(testloader):
            # current image before current action
            x = batch_data['image_bi_cur']
            x = x.float().to(device).view(-1, 1, 50, 50)
            # current action
            action = batch_data['resz_action_cur']
            action = action.float().to(device).view(-1, 4)
            # post image after current action
            x_next = batch_data['image_bi_post']
            x_next = x_next.float().to(device).view(-1, 1, 50, 50) 
            # model
            x_dec, x_next_pred_dec, Qz, Qz_next, Qz_next_pred = e2c_model(x, action, x_next)
            # prediction
            x_next_pred = e2c_model.predict(x, action)
            # loss
            loss_bound, loss_kl = compute_loss(x_dec, x_next_pred_dec, x, x_next, Qz, Qz_next_pred, Qz_next)
            loss_pred = F.binary_cross_entropy(x_next_pred.view(-1, 2500), x_next.view(-1, 2500), reduction='sum')
            loss = GAMMA_bound * loss_bound + GAMMA_kl * loss_kl + GAMMA_pred * loss_pred
            test_loss += loss.item()
            bound_loss += GAMMA_bound * loss_bound.item()
            kl_loss += GAMMA_kl * loss_kl.item()
            pred_loss += GAMMA_pred * loss_pred.item()
            if batch_idx == 0:
                n = min(batch_data['image_bi_cur'].size(0), 8)
                comparison = torch.cat([batch_data['image_bi_cur'][:n],                 # current image
                                      x_dec.view(-1, 1, 50, 50).cpu()[:n],      # reconstruction of current image
                                      batch_data['image_bi_post'][:n],                  # post image
                                      x_next_pred.view(-1, 1, 50, 50).cpu()[:n]])     # prediction of post image
                save_image(comparison.cpu(),
                         './result/{}/reconstruction_test/reconstruct_epoch_{}.png'.format(folder_name, epoch), nrow=n)  
    n = len(testloader.dataset)
    return test_loss/n, bound_loss/n, kl_loss/n, pred_loss/n

# args
parser = argparse.ArgumentParser(description='E2C Rope Deform Example')
parser.add_argument('--folder-name', default='test_E2C', 
                    help='set folder name to save image files')#folder_name = 'test_new_train_scale_large'
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 500)')   
parser.add_argument('--gamma-kl', type=int, default=1, metavar='N',
                    help='scale coefficient for loss of kl divergence for z (default: 10)')   
parser.add_argument('--gamma-pred', type=int, default=1, metavar='N',
                    help='scale coefficient for loss of prediction (default: 100)')                                                                           
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')                   
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--restore', action='store_true', default=False)                    
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

          
# dataset
print('***** Preparing Data *****')
total_img_num = 100#22515
train_num = int(total_img_num * 0.8)
image_paths_bi = create_image_path('rope_no_loop_all_resize_gray_clean', total_img_num)
#image_paths_ori = create_image_path('rope_all_ori', total_img_num)
resz_act_path = './rope_dataset/rope_no_loop_all_resize_gray_clean/simplified_clean_actions_all_size50.npy'
#ori_act_path = './rope_dataset/rope_all_ori/actions.npy'
resz_act = np.load(resz_act_path)
#ori_act = np.load(ori_act_path)
# transform = transforms.Compose([Translation(10), 
#                                 HFlip(0.5), 
#                                 VFlip(0.5), 
#                                 ToTensor()])   
transform = transforms.Compose([Translation(10), 
                                ToTensor()])                           
#dataset = MyDataset(image_paths_bi, resz_act, transform=ToTensor())
trainset = MyDataset(image_paths_bi[0:train_num], resz_act[0:train_num], transform=transform)
testset = MyDataset(image_paths_bi[train_num:], resz_act[train_num:], transform=ToTensor())
trainloader = DataLoader(trainset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, collate_fn=my_collate)
testloader = DataLoader(testset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, collate_fn=my_collate)                        
print('***** Finish Preparing Data *****')


# train var
GAMMA_kl = args.gamma_kl
GAMMA_pred = args.gamma_pred

# create folders
folder_name = args.folder_name
create_folder(folder_name)

print('***** Start Training & Testing *****')
device = torch.device("cuda" if args.cuda else "cpu")
epochs = args.epochs
e2c_model = E2C().to(device)
e2c_optimizer = optim.Adam(e2c_model.parameters(), lr=1e-3)

# initial train
if not args.restore:
    init_epoch = 1
    loss_logger = None
# restore previous train        
else:
    print('***** Load Checkpoint *****')
    PATH = './result/{}/checkpoint'.format(folder_name)
    checkpoint = torch.load(PATH, map_location=device)
    e2c_model.load_state_dict(checkpoint['e2c_model_state_dict'])  
    e2c_optimizer.load_state_dict(checkpoint['e2c_optimizer_state_dict'])
    init_epoch = checkpoint['epoch'] + 1
    loss_logger = checkpoint['loss_logger'] 

train_loss_all = []
test_loss_all = []
train_bound_loss_all = []
test_bound_loss_all = []
train_kl_loss_all = []
test_kl_loss_all = []
train_pred_loss_all = []
test_pred_loss_all = []
for epoch in range(init_epoch, epochs+1):
    train_loss, train_bound_loss, train_kl_loss, train_pred_loss = train(e2c_model)
    test_loss, test_bound_loss, test_kl_loss, test_pred_loss = test(e2c_model)
    train_loss_all.append(train_loss)
    test_loss_all.append(test_loss)
    train_bound_loss_all.append(train_bound_loss)
    test_bound_loss_all.append(test_bound_loss)
    train_kl_loss_all.append(train_kl_loss)
    test_kl_loss_all.append(test_kl_loss)
    train_pred_loss_all.append(train_pred_loss)
    test_pred_loss_all.append(test_pred_loss)
    if epoch % args.log_interval == 0:
        save_e2c_data(folder_name, epochs, train_loss_all, test_loss_all)        
        # save checkpoint
        PATH = './result/{}/checkpoint'.format(folder_name)
        loss_logger = {'train_loss_all': train_loss_all, 'train_bound_loss_all': train_bound_loss_all,
                       'train_kl_loss_all': train_kl_loss_all, 'train_pred_loss_all': train_pred_loss_all, 
                       'test_loss_all': test_loss_all, 'test_bound_loss_all': test_bound_loss_all, 
                       'test_kl_loss_all': test_kl_loss_all, 'test_pred_loss_all': test_pred_loss_all}
        torch.save({
                    'epoch': epoch,
                    'e2c_model_state_dict': e2c_model.state_dict(),
                    'e2c_optimizer_state_dict': e2c_optimizer.state_dict(),
                    'loss_logger': loss_logger
                    }, 
                    PATH)

save_e2c_data(folder_name, epochs, train_loss_all, train_bound_loss_all, train_kl_loss_all, train_pred_loss_all, \
              test_loss_all, test_bound_loss_all, test_kl_loss_all, test_pred_loss_all)  

# plot
plot_train_loss('./result/{}/train_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_train_bound_loss('./result/{}/train_bound_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_train_kl_loss('./result/{}/train_kl_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_train_pred_loss('./result/{}/train_pred_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_test_loss('./result/{}/test_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_test_bound_loss('./result/{}/test_bound_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_test_kl_loss('./result/{}/test_kl_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)
plot_test_pred_loss('./result/{}/test_pred_loss_epoch{}.npy'.format(folder_name, epochs), folder_name)

# save checkpoint
PATH = './result/{}/checkpoint'.format(folder_name)
loss_logger = {'train_loss_all': train_loss_all, 'test_loss_all': test_loss_all}
torch.save({
            'epoch': epoch,
            'e2c_model_state_dict': e2c_model.state_dict(),
            'e2c_optimizer_state_dict': e2c_optimizer.state_dict(),
            'loss_logger': loss_logger
            }, 
            PATH)
print('***** End Program *****')  