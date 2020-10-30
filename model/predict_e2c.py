from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from torch import nn, optim, sigmoid, tanh, relu
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from deform.model.create_dataset import *
from deform.model.hidden_dynamics import *
from deform.utils.utils import plot_sample_multi_step
from torchvision.utils import save_image
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
        v1 = v.unsqueeze(2)
        rT = r.unsqueeze(1)
        I = Variable(torch.eye(self.dim_z).repeat(batch_size, 1, 1))
        if rT.data.is_cuda:
            I.dada.cuda()
        A = I.add(v1.bmm(rT))

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
        if std.data.is_cuda:
            eps.cuda()
        eps = Variable(eps)
        return eps.mul(std).add_(mean), NormalDistribution(mean, std, torch.log(std))

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

def predict():
    e2c_model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # order: img_pre -> act_pre -> img_cur -> act_cur -> img_post
            # previous image 
            img_pre = batch_data['image_bi_pre']
            img_pre = img_pre.float().to(device).view(-1, 1, 50, 50)
            # previous action
            act_pre = batch_data['resz_action_pre']
            act_pre = act_pre.float().to(device).view(-1, 4)
            # current image 
            img_cur = batch_data['image_bi_cur']
            img_cur = img_cur.float().to(device).view(-1, 1, 50, 50) 
            # current action
            act_cur = batch_data['resz_action_cur']
            act_cur = act_cur.float().to(device).view(-1, 4)
            # post image
            img_post = batch_data['image_bi_post']
            img_post = img_post.float().to(device).view(-1, 1, 50, 50)  
            # post action
            act_post = batch_data['resz_action_post']
            act_post = act_post.float().to(device).view(-1, 4)
            # post2 image
            img_post2 = batch_data['image_bi_post2']
            img_post2 = img_post2.float().to(device).view(-1, 1, 50, 50)  
            # post2 action
            act_post2 = batch_data['resz_action_post2']
            act_post2 = act_post2.float().to(device).view(-1, 4)
            # post3 image
            img_post3 = batch_data['image_bi_post3']
            img_post3 = img_post3.float().to(device).view(-1, 1, 50, 50)  
            # post3 action
            act_post3 = batch_data['resz_action_post3']
            act_post3 = act_post3.float().to(device).view(-1, 4)
            # post4 image
            img_post4 = batch_data['image_bi_post4']
            img_post4 = img_post4.float().to(device).view(-1, 1, 50, 50) 
            # post4 action
            act_post4 = batch_data['resz_action_post4']
            act_post4 = act_post4.float().to(device).view(-1, 4)
            # post5 image
            img_post5 = batch_data['image_bi_post5']
            img_post5 = img_post5.float().to(device).view(-1, 1, 50, 50) 
            # post5 action
            act_post5 = batch_data['resz_action_post5']
            act_post5 = act_post5.float().to(device).view(-1, 4)
            # post6 image
            img_post6 = batch_data['image_bi_post6']
            img_post6 = img_post6.float().to(device).view(-1, 1, 50, 50) 
            # post6 action
            act_post6 = batch_data['resz_action_post6']
            act_post6 = act_post6.float().to(device).view(-1, 4)
            # post7 image
            img_post7 = batch_data['image_bi_post7']
            img_post7 = img_post7.float().to(device).view(-1, 1, 50, 50) 
            # post7 action
            act_post7 = batch_data['resz_action_post7']
            act_post7 = act_post7.float().to(device).view(-1, 4)
            # post8 image
            img_post8 = batch_data['image_bi_post8']
            img_post8 = img_post8.float().to(device).view(-1, 1, 50, 50) 
            # post8 action
            act_post8 = batch_data['resz_action_post8']
            act_post8 = act_post8.float().to(device).view(-1, 4)
            # post9 image
            img_post9 = batch_data['image_bi_post9']
            img_post9 = img_post9.float().to(device).view(-1, 1, 50, 50)                                                                                       
            # ten step prediction            
            # prediction for current image from pre image
            recon_img_cur = e2c_model.predict(img_pre, act_pre)
            # prediction for post image from pre image
            recon_img_post = e2c_model.predict(recon_img_cur, act_cur)       
            # prediction for post2 image from pre image
            recon_img_post2 = e2c_model.predict(recon_img_post, act_post)   
            # prediction for post3 image from pre image
            recon_img_post3 = e2c_model.predict(recon_img_post2, act_post2)   
            # prediction for post4 image from pre image
            recon_img_post4 = e2c_model.predict(recon_img_post3, act_post3) 
            # prediction for post5 image from pre image
            recon_img_post5 = e2c_model.predict(recon_img_post4, act_post4) 
            # prediction for post6 image from pre image
            recon_img_post6 = e2c_model.predict(recon_img_post5, act_post5) 
            # prediction for post7 image from pre image
            recon_img_post7 = e2c_model.predict(recon_img_post6, act_post6) 
            # prediction for post8 image from pre image
            recon_img_post8 = e2c_model.predict(recon_img_post7, act_post7) 
            # prediction for post9 image from pre image
            recon_img_post9 = e2c_model.predict(recon_img_post8, act_post8)                                                                                            
            if batch_idx % 10 == 0:
                n = min(batch_data['image_bi_pre'].size(0), 1)
                comparison_GT = torch.cat([batch_data['image_bi_pre'][:n],
                                        batch_data['image_bi_cur'][:n],
                                        batch_data['image_bi_post'][:n],
                                        batch_data['image_bi_post2'][:n],
                                        batch_data['image_bi_post3'][:n],
                                        batch_data['image_bi_post4'][:n],
                                        batch_data['image_bi_post5'][:n],
                                        batch_data['image_bi_post6'][:n],
                                        batch_data['image_bi_post7'][:n],
                                        batch_data['image_bi_post8'][:n],
                                        batch_data['image_bi_post9'][:n]])                                        
                save_image(comparison_GT.cpu(),
                         './result/{}/prediction_full_step{}/prediction_GT_batch{}.png'.format(folder_name, step, batch_idx), nrow=n)                                         
                comparison_Pred = torch.cat([batch_data['image_bi_pre'][:n],
                                        recon_img_cur.view(-1, 1, 50, 50).cpu()[:n],
                                        recon_img_post.view(-1, 1, 50, 50).cpu()[:n],
                                        recon_img_post2.view(-1, 1, 50, 50).cpu()[:n],
                                        recon_img_post3.view(-1, 1, 50, 50).cpu()[:n],
                                        recon_img_post4.view(-1, 1, 50, 50).cpu()[:n],
                                        recon_img_post5.view(-1, 1, 50, 50).cpu()[:n],
                                        recon_img_post6.view(-1, 1, 50, 50).cpu()[:n],
                                        recon_img_post7.view(-1, 1, 50, 50).cpu()[:n],
                                        recon_img_post8.view(-1, 1, 50, 50).cpu()[:n],
                                        recon_img_post9.view(-1, 1, 50, 50).cpu()[:n]])                                        
                save_image(comparison_Pred.cpu(),
                         './result/{}/prediction_full_step{}/prediction_Pred_batch{}.png'.format(folder_name, step, batch_idx), nrow=n) 
                #GT
                plot_sample_multi_step(batch_data['image_bi_pre'][:n].detach().cpu().numpy(), 
                            batch_data['image_bi_cur'][:n].detach().cpu().numpy(), 
                            batch_data['image_bi_post'][:n].detach().cpu().numpy(), 
                            batch_data['image_bi_post2'][:n].detach().cpu().numpy(), 
                            batch_data['image_bi_post3'][:n].detach().cpu().numpy(), 
                            batch_data['image_bi_post4'][:n].detach().cpu().numpy(), 
                            batch_data['image_bi_post5'][:n].detach().cpu().numpy(), 
                            batch_data['image_bi_post6'][:n].detach().cpu().numpy(), 
                            batch_data['image_bi_post7'][:n].detach().cpu().numpy(), 
                            batch_data['image_bi_post8'][:n].detach().cpu().numpy(), 
                            batch_data['image_bi_post9'][:n].detach().cpu().numpy(), 
                            batch_data['resz_action_pre'][:n].detach().cpu().numpy(),
                            batch_data['resz_action_cur'][:n].detach().cpu().numpy(), 
                            batch_data['resz_action_post'][:n].detach().cpu().numpy(),
                            batch_data['resz_action_post2'][:n].detach().cpu().numpy(),
                            batch_data['resz_action_post3'][:n].detach().cpu().numpy(),
                            batch_data['resz_action_post4'][:n].detach().cpu().numpy(),
                            batch_data['resz_action_post5'][:n].detach().cpu().numpy(),
                            batch_data['resz_action_post6'][:n].detach().cpu().numpy(),
                            batch_data['resz_action_post7'][:n].detach().cpu().numpy(),
                            batch_data['resz_action_post8'][:n].detach().cpu().numpy(),
                            './result/{}/prediction_with_action_step{}/recon_GT_epoch_{}.png'.format(folder_name, step, batch_idx))  
                # Predicted
                plot_sample_multi_step(batch_data['image_bi_pre'][:n].detach().cpu().numpy(),       
                            recon_img_cur.view(-1, 1, 50, 50)[:n].detach().cpu().numpy(),                   
                            recon_img_post.view(-1, 1, 50, 50)[:n].detach().cpu().numpy(), 
                            recon_img_post2.view(-1, 1, 50, 50)[:n].detach().cpu().numpy(), 
                            recon_img_post3.view(-1, 1, 50, 50)[:n].detach().cpu().numpy(), 
                            recon_img_post4.view(-1, 1, 50, 50)[:n].detach().cpu().numpy(), 
                            recon_img_post5.view(-1, 1, 50, 50)[:n].detach().cpu().numpy(), 
                            recon_img_post6.view(-1, 1, 50, 50)[:n].detach().cpu().numpy(), 
                            recon_img_post7.view(-1, 1, 50, 50)[:n].detach().cpu().numpy(), 
                            recon_img_post8.view(-1, 1, 50, 50)[:n].detach().cpu().numpy(), 
                            recon_img_post9.view(-1, 1, 50, 50)[:n].detach().cpu().numpy(),  
                            batch_data['resz_action_pre'][:n].detach().cpu().numpy(),
                            batch_data['resz_action_cur'][:n].detach().cpu().numpy(),
                            batch_data['resz_action_post'][:n].detach().cpu().numpy(),
                            batch_data['resz_action_post2'][:n].detach().cpu().numpy(),
                            batch_data['resz_action_post3'][:n].detach().cpu().numpy(),
                            batch_data['resz_action_post4'][:n].detach().cpu().numpy(),
                            batch_data['resz_action_post5'][:n].detach().cpu().numpy(),
                            batch_data['resz_action_post6'][:n].detach().cpu().numpy(),
                            batch_data['resz_action_post7'][:n].detach().cpu().numpy(),
                            batch_data['resz_action_post8'][:n].detach().cpu().numpy(),
                            './result/{}/prediction_with_action_step{}/recon_Pred_epoch_{}.png'.format(folder_name, step, batch_idx))
                            
print('***** Preparing Data *****')
total_img_num = 200#22515
image_paths_bi = create_image_path('rope_no_loop_all_resize_gray_clean', total_img_num)
action_path = './rope_dataset/rope_no_loop_all_resize_gray_clean/simplified_clean_actions_all_size50.npy'
actions = np.load(action_path)
dataset = MyDatasetMultiPred10(image_paths_bi, actions, transform=ToTensorMultiPred10())   
dataloader = DataLoader(dataset, batch_size=32, # batch size 32 to 16
                        shuffle=True, num_workers=4, collate_fn=my_collate)    # num_workers 4->1                                         
print('***** Finish Preparing Data *****')

folder_name = 'test_E2C'
PATH = './result/{}/checkpoint'.format(folder_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
e2c_model = E2C().to(device)


# load check point
print('***** Load Checkpoint *****')
checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
e2c_model.load_state_dict(checkpoint['e2c_model_state_dict'])  

# prediction
print('***** Start Prediction *****')
step=10 # Change this based on different prediction steps
if not os.path.exists('./result/{}/prediction_full_step{}'.format(folder_name, step)):
    os.makedirs('./result/{}/prediction_full_step{}'.format(folder_name, step))
if not os.path.exists('./result/{}/prediction_with_action_step{}'.format(folder_name, step)):
    os.makedirs('./result/{}/prediction_with_action_step{}'.format(folder_name, step))    
predict()
print('***** Finish Prediction *****')