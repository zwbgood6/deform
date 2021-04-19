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

class CAE(nn.Module):
    def __init__(self, latent_state_dim=80, latent_act_dim=80):
        super(CAE, self).__init__()
        # state
        self.conv_layers = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),  
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2),
                                         nn.Conv2d(32, 64, 3, padding=1), 
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2),
                                         nn.Conv2d(64, 128, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2),
                                         nn.Conv2d(128, 128, 3, padding=1), 
                                         nn.ReLU(),
                                         nn.Conv2d(128, 128, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2, padding=1))  
        self.fc1 = nn.Linear(128*3*3, latent_state_dim)
        self.fc2 = nn.Linear(latent_state_dim, 128*3*3)       
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
        # action
        self.fc5 = nn.Linear(4, latent_act_dim)
        self.fc6 = nn.Linear(latent_act_dim, latent_act_dim) 
        self.fc7 = nn.Linear(latent_act_dim, latent_act_dim) 
        self.fc8 = nn.Linear(latent_act_dim, 4)  
        # add these in order to use GPU for parameters
        self.mul_tensor = torch.tensor([50, 50, 2*math.pi, 0.14]) 
        self.add_tensor = torch.tensor([0, 0, 0, 0.01]) 


    def encoder(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1) 
        return relu(self.fc1(x))

    def decoder(self, x):
        x = relu(self.fc2(x))
        x = x.view(-1, 128, 3, 3) 
        return self.dconv_layers(x)

    def encoder_act(self, u):
        h1 = relu(self.fc5(u))
        return relu(self.fc6(h1))

    def decoder_act(self, u):
        h2 = relu(self.fc7(u))
        if torch.cuda.is_available():
            return torch.mul(sigmoid(self.fc8(h2)), self.mul_tensor.cuda()) + self.add_tensor.cuda()
        else:
            return torch.mul(sigmoid(self.fc8(h2)), self.mul_tensor) + self.add_tensor


    def forward(self, x_cur, u, x_post):
        g_cur = self.encoder(x_cur) 
        a = self.encoder_act(u)  
        g_post = self.encoder(x_post)     
       
        return g_cur, a, g_post, self.decoder(g_cur), self.decoder_act(a)

class SysDynamics(nn.Module):
    def __init__(self, latent_state_dim=80, latent_act_dim=80):
        super(SysDynamics, self).__init__()
        self.conv_layers_matrix = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),  
                                            nn.ReLU(),
                                            nn.MaxPool2d(3, stride=1),
                                            nn.Conv2d(32, 64, 3, padding=1),  
                                            nn.ReLU(),
                                            nn.MaxPool2d(3, stride=1),                                            
                                            nn.Conv2d(64, 128, 3, padding=1), 
                                            nn.ReLU(),
                                            nn.MaxPool2d(3, stride=2),
                                            nn.Conv2d(128, 256, 3, padding=1), 
                                            nn.ReLU(),
                                            nn.MaxPool2d(3, stride=2),
                                            nn.Conv2d(256, 512, 3, padding=1), 
                                            nn.ReLU(),
                                            nn.MaxPool2d(3, stride=2),                                            
                                            nn.Conv2d(512, 512, 3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(512, 512, 3, padding=1),
                                            nn.ReLU(),                                            
                                            nn.MaxPool2d(3, stride=2, padding=1)) 
        self.fc31 = nn.Linear(512*2*2, latent_state_dim*latent_state_dim)
        self.fc32 = nn.Linear(latent_state_dim*latent_state_dim, latent_state_dim*latent_state_dim)
        self.fc41 = nn.Linear(512*2*2 + latent_act_dim, latent_state_dim*latent_act_dim) 
        self.fc42 = nn.Linear(latent_state_dim*latent_act_dim, latent_state_dim*latent_act_dim)     
        self.fc9 = nn.Linear(4, latent_act_dim)
        self.fc10 = nn.Linear(latent_act_dim, latent_act_dim)

        self.latent_act_dim = latent_act_dim
        self.latent_state_dim = latent_state_dim

    def encoder_matrix(self, x, a):
        x = self.conv_layers_matrix(x) 
        x = x.view(x.shape[0], -1)
        xa = torch.cat((x,a), 1)

        return relu(self.fc32(relu(self.fc31(x)))).view(-1, self.latent_state_dim, self.latent_state_dim), \
            relu(self.fc42(relu(self.fc41(xa)))).view(-1, self.latent_act_dim, self.latent_state_dim)

    def forward(self, x_cur, u):
        a = relu(self.fc10(relu(self.fc9(u))))  
        K_T, L_T = self.encoder_matrix(x_cur, a) 
       
        return K_T, L_T

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
                                         nn.Conv2d(128, 128, 3, padding=1), 
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
        # action
        self.fc5 = nn.Linear(4, dim_u*20)
        self.fc6 = nn.Linear(dim_u*20, dim_u) 
        self.fc7 = nn.Linear(dim_u, dim_u*20) 
        self.fc8 = nn.Linear(dim_u*20, 4) 
        self.mul_tensor = torch.tensor([50, 50, 2*math.pi, 0.14]) 
        self.add_tensor = torch.tensor([0, 0, 0, 0.01]) 

    def encode(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1) 
        return relu(self.fc1(x)).chunk(2, dim=1)

    def decode(self, x):
        x = relu(self.fc2(x))
        x = x.view(-1, 128, 3, 3)
        return self.dconv_layers(x)

    def encode_act(self, u):
        h1 = relu(self.fc5(u))
        return relu(self.fc6(h1))

    def decode_act(self, u):
        h2 = relu(self.fc7(u))
        return torch.mul(sigmoid(self.fc8(h2)), self.mul_tensor.cuda()) + self.add_tensor.cuda() 

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

        u = u.unsqueeze(2)

        d = A.bmm(Q.mu.unsqueeze(2)).add(B.bmm(u)).add(o).squeeze(2)
        sample = A.bmm(h.unsqueeze(2)).add(B.bmm(u)).add(o).squeeze(2)        
        return sample, NormalDistribution(d, Q.sigma, Q.logsigma, v=v, r=r)

    def reparam(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        self.z_mean = mean
        self.z_sigma = std
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std.cpu()).add_(mean.cpu()).cuda(), NormalDistribution(mean, std, torch.log(std))

    def forward(self, x, action, x_next):
        mean, logvar = self.encode(x)
        mean_next, logvar_next = self.encode(x_next)

        z, self.Qz = self.reparam(mean, logvar)
        z_next, self.Qz_next = self.reparam(mean_next, logvar_next)

        self.x_dec = self.decode(z)
        self.x_next_dec = self.decode(z_next)

        latent_a = self.encode_act(action)
        action_dec = self.decode_act(latent_a)

        self.z_next_pred, self.Qz_next_pred = self.transition(z, self.Qz, latent_a)
        self.x_next_pred_dec = self.decode(self.z_next_pred)

        
        return self.x_dec, self.x_next_dec, self.x_next_pred_dec, self.Qz, self.Qz_next, self.Qz_next_pred, action_dec

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
    recon_model.eval()
    dyn_model.eval()    
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
            if batch_idx % 5 == 0:
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
                         './result/{}/prediction_full_step{}/prediction_GT_batch{}.png'.format(folder_name_e2c, step, batch_idx), nrow=n)                                         
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
                         './result/{}/prediction_full_step{}/prediction_Pred_batch{}.png'.format(folder_name_e2c, step, batch_idx), nrow=n) 
            # ten step prediction            
            # prediction for current image from pre image
            latent_img_pre, latent_act_pre, _, _, _ = recon_model(img_pre, act_pre, img_cur)
            K_T_pre, L_T_pre = dyn_model(img_pre, act_pre)
            recon_latent_img_cur = get_next_state_linear(latent_img_pre, latent_act_pre, K_T_pre, L_T_pre)
            recon_img_cur = recon_model.decoder(recon_latent_img_cur)
            # prediction for post image from pre image
            _, latent_act_cur, _, _, _ = recon_model(img_cur, act_cur, img_post)
            K_T_cur, L_T_cur = dyn_model(recon_img_cur, act_cur)
            recon_latent_img_post = get_next_state_linear(recon_latent_img_cur, latent_act_cur, K_T_cur, L_T_cur)
            recon_img_post = recon_model.decoder(recon_latent_img_post)            
            # prediction for post2 image from pre image
            _, latent_act_post, _, _, _ = recon_model(img_post, act_post, img_post2)
            K_T_post, L_T_post = dyn_model(recon_img_post, act_post)
            recon_latent_img_post2 = get_next_state_linear(recon_latent_img_post, latent_act_post, K_T_post, L_T_post)
            recon_img_post2 = recon_model.decoder(recon_latent_img_post2) 
            # prediction for post3 image from pre image
            _, latent_act_post2, _, _, _ = recon_model(img_post2, act_post2, img_post3)
            K_T_post2, L_T_post2 = dyn_model(recon_img_post2, act_post2)
            recon_latent_img_post3 = get_next_state_linear(recon_latent_img_post2, latent_act_post2, K_T_post2, L_T_post2)
            recon_img_post3 = recon_model.decoder(recon_latent_img_post3)  
            # prediction for post4 image from pre image
            _, latent_act_post3, _, _, _ = recon_model(img_post3, act_post3, img_post4)
            K_T_post3, L_T_post3 = dyn_model(recon_img_post3, act_post3)
            recon_latent_img_post4 = get_next_state_linear(recon_latent_img_post3, latent_act_post3, K_T_post3, L_T_post3)
            recon_img_post4 = recon_model.decoder(recon_latent_img_post4) 
            # prediction for post5 image from pre image
            _, latent_act_post4, _, _, _ = recon_model(img_post4, act_post4, img_post5)
            K_T_post4, L_T_post4 = dyn_model(recon_img_post4, act_post4)
            recon_latent_img_post5 = get_next_state_linear(recon_latent_img_post4, latent_act_post4, K_T_post4, L_T_post4)
            recon_img_post5 = recon_model.decoder(recon_latent_img_post5) 
            # prediction for post6 image from pre image
            _, latent_act_post5, _, _, _ = recon_model(img_post5, act_post5, img_post6)
            K_T_post5, L_T_post5 = dyn_model(recon_img_post5, act_post5)
            recon_latent_img_post6 = get_next_state_linear(recon_latent_img_post5, latent_act_post5, K_T_post5, L_T_post5)
            recon_img_post6 = recon_model.decoder(recon_latent_img_post6)
            # prediction for post7 image from pre image
            _, latent_act_post6, _, _, _ = recon_model(img_post6, act_post6, img_post7)
            K_T_post6, L_T_post6 = dyn_model(recon_img_post6, act_post6)
            recon_latent_img_post7 = get_next_state_linear(recon_latent_img_post6, latent_act_post6, K_T_post6, L_T_post6)
            recon_img_post7 = recon_model.decoder(recon_latent_img_post7)
            # prediction for post8 image from pre image
            _, latent_act_post7, _, _, _ = recon_model(img_post7, act_post7, img_post8)
            K_T_post7, L_T_post7 = dyn_model(recon_img_post7, act_post7)
            recon_latent_img_post8 = get_next_state_linear(recon_latent_img_post7, latent_act_post7, K_T_post7, L_T_post7)
            recon_img_post8 = recon_model.decoder(recon_latent_img_post8)
            # prediction for post9 image from pre image
            _, latent_act_post8, _, _, _ = recon_model(img_post8, act_post8, img_post9)
            K_T_post8, L_T_post8 = dyn_model(recon_img_post8, act_post8)
            recon_latent_img_post9 = get_next_state_linear(recon_latent_img_post8, latent_act_post8, K_T_post8, L_T_post8)
            recon_img_post9 = recon_model.decoder(recon_latent_img_post9)                                                                                             
            if batch_idx % 5 == 0:
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
                         './result/{}/prediction_full_step{}/prediction_GT_batch{}.png'.format(folder_name_our, step, batch_idx), nrow=n)                                         
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
                         './result/{}/prediction_full_step{}/prediction_Pred_batch{}.png'.format(folder_name_our, step, batch_idx), nrow=n) 
                            
print('***** Preparing Data *****')
total_img_num = 22515
image_paths_bi = create_image_path('rope_no_loop_all_resize_gray_clean', total_img_num)
action_path = './rope_dataset/rope_no_loop_all_resize_gray_clean/simplified_clean_actions_all_size50.npy'
actions = np.load(action_path)
dataset = MyDatasetMultiPred10(image_paths_bi, actions, transform=ToTensorMultiPred10())   
dataloader = DataLoader(dataset, batch_size=32, 
                        shuffle=True, num_workers=4, collate_fn=my_collate)                               
print('***** Finish Preparing Data *****')

folder_name_e2c = 'test_E2C_gpu_update_loss'
PATH_e2c = './result/{}/checkpoint'.format(folder_name_e2c)
folder_name_our = 'test_act80_pred30'
PATH_our = './result/{}/checkpoint'.format(folder_name_our)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
e2c_model = E2C().to(device)
recon_model = CAE().to(device)
dyn_model = SysDynamics().to(device)

# load check point
print('***** Load Checkpoint *****')
checkpoint_e2c = torch.load(PATH_e2c, map_location=torch.device('cpu'))
e2c_model.load_state_dict(checkpoint_e2c['e2c_model_state_dict'])  
checkpoint_our = torch.load(PATH_our, map_location=torch.device('cpu'))
recon_model.load_state_dict(checkpoint_our['recon_model_state_dict'])
dyn_model.load_state_dict(checkpoint_our['dyn_model_state_dict'])

# prediction
print('***** Start Prediction *****')
step=10 # Change this based on different prediction steps
if not os.path.exists('./result/{}/prediction_full_step{}'.format(folder_name_e2c, step)):
    os.makedirs('./result/{}/prediction_full_step{}'.format(folder_name_e2c, step))
if not os.path.exists('./result/{}/prediction_with_action_step{}'.format(folder_name_e2c, step)):
    os.makedirs('./result/{}/prediction_with_action_step{}'.format(folder_name_e2c, step))    
if not os.path.exists('./result/{}/prediction_full_step{}'.format(folder_name_our, step)):
    os.makedirs('./result/{}/prediction_full_step{}'.format(folder_name_our, step))
if not os.path.exists('./result/{}/prediction_with_action_step{}'.format(folder_name_our, step)):
    os.makedirs('./result/{}/prediction_with_action_step{}'.format(folder_name_our, step))  
predict()
print('***** Finish Prediction *****')
