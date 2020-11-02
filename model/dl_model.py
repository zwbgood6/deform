import torch
from torch import nn, optim, sigmoid, tanh, relu
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
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
        # action
        self.fc5 = nn.Linear(4, dim_u*20)
        self.fc6 = nn.Linear(dim_u*20, dim_u) 
        self.fc7 = nn.Linear(dim_u, dim_u*20) # 10-100
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
        a = torch.cat((h, h))
        v, r = self.trans(a).chunk(2, dim=1)
        v = v[0].reshape((1, -1))
        r = r[0].reshape((1, -1))
        v1 = v.unsqueeze(2).cpu()
        rT = r.unsqueeze(1).cpu()
        I = Variable(torch.eye(self.dim_z).repeat(batch_size, 1, 1))
        if rT.data.is_cuda:
            I.data.cuda()
        A = I.add(v1.bmm(rT))

        B = self.fc_B(h).view(-1, self.dim_z, self.dim_u)
        o = self.fc_o(h).reshape((-1, self.dim_z, 1))

        # need to compute the parameters for distributions
        # as well as for the samples
        u = u.reshape((1, -1))
        u = u.unsqueeze(2)
        u = u.type(torch.FloatTensor)
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
        return eps.mul(std.cpu()).add_(mean.cpu()), NormalDistribution(mean, std, torch.log(std))

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
                                         nn.Conv2d(128, 128, 3, padding=1), # channel 1 32 64 64; the next batch size should be larger than 8, 4 corner features + 4 direction features
                                         nn.ReLU(),
                                         nn.Conv2d(128, 128, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(3, stride=2, padding=1))  
        self.fc1 = nn.Linear(128*3*3, latent_state_dim) # size: 128*3*3 > latent_state_dim
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
        self.fc7 = nn.Linear(latent_act_dim, latent_act_dim) # 10-100
        self.fc8 = nn.Linear(latent_act_dim, 4)  
        # add these in order to use GPU for parameters
        self.mul_tensor = torch.tensor([50, 50, 2*math.pi, 0.14]) 
        self.add_tensor = torch.tensor([0, 0, 0, 0.01]) 
        # latent dim
        #self.latent_act_dim = latent_act_dim
        #self.latent_state_dim = latent_state_dim

    def encoder(self, x):
        if x is None:
            return None
        else:            
            x = self.conv_layers(x)
            x = x.view(x.shape[0], -1) 
            return relu(self.fc1(x))

    def decoder(self, x):
        x = relu(self.fc2(x))
        x = x.view(-1, 128, 3, 3) #(batch size, channel, H, W)
        return self.dconv_layers(x)

    def encoder_act(self, u):
        h1 = relu(self.fc5(u))
        return relu(self.fc6(h1))

    def decoder_act(self, u):
        h2 = relu(self.fc7(u))
        return torch.mul(sigmoid(self.fc8(h2)), self.mul_tensor) + self.add_tensor

    def forward(self, x_cur, u, x_post):
        # print('x_cur shape', x_cur.shape)
        g_cur = self.encoder(x_cur) 
        # print('g_cur shape', g_cur.shape)
        # print('u shape', u.shape)
        a = self.encoder_act(u)  
        # print('a shape', a.shape)
        g_post = self.encoder(x_post)     
        # print('x_cur shape', x_cur.shape)
        # print('a shape', a.shape)
        # print('x_post shape', x_post.shape)
        #K_T, L_T = self.encoder_matrix(x_cur, a) 
        # print('K_T shape', K_T.shape) 
        # print('L_T shape', L_T.shape)        
        return g_cur, a, g_post, self.decoder(g_cur), self.decoder_act(a)#, K_T, L_T#self.control_matrix

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
        self.fc31 = nn.Linear(512*2*2, latent_state_dim*latent_state_dim) # K: 9216 -> 6400
        self.fc32 = nn.Linear(latent_state_dim*latent_state_dim, latent_state_dim*latent_state_dim) # K: 9216 -> 6400
        #self.fc32 = nn.Linear(3000, latent_state_dim*latent_state_dim) 
        self.fc41 = nn.Linear(512*2*2 + latent_act_dim, latent_state_dim*latent_act_dim) # L: 9216+40 -> 3200  
        self.fc42 = nn.Linear(latent_state_dim*latent_act_dim, latent_state_dim*latent_act_dim) # L: 9216+40 -> 3200       
        self.fc9 = nn.Linear(4, latent_act_dim)
        self.fc10 = nn.Linear(latent_act_dim, latent_act_dim)
        # latent dim
        self.latent_act_dim = latent_act_dim
        self.latent_state_dim = latent_state_dim

    def encoder_matrix(self, x, a):
        # print('x_cur shape', x_cur.shape)
        # print('x_post shape', x_post.shape)
        #x = torch.cat((x_cur, x_post), 1)
        # print('after concatenation shape', x.shape)
        x = self.conv_layers_matrix(x) # output size: 256*6*6
        # print('after convolution shape', x.shape)
        x = x.view(x.shape[0], -1)
        #print('x shape', x.shape)
        xa = torch.cat((x,a), 1)
        #print('xu shape', xa.shape)
        return relu(self.fc32(relu(self.fc31(x)))).view(-1, self.latent_state_dim, self.latent_state_dim), \
            relu(self.fc42(relu(self.fc41(xa)))).view(-1, self.latent_act_dim, self.latent_state_dim)

    def forward(self, x_cur, u):
        a = relu(self.fc10(relu(self.fc9(u))))  
        K_T, L_T = self.encoder_matrix(x_cur, a) 
        # print('K_T shape', K_T.shape) 
        # print('L_T shape', L_T.shape)        
        return K_T, L_T#self.control_matrix