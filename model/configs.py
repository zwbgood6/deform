"""
Configuration for the encoder, decoder, transition
for different tasks. Use load_config to find the proper
set of configuration.
"""
import torch
from torch import nn
from torch.autograd import Variable

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


class Encoder(nn.Module):
    def __init__(self, enc, dim_in, dim_out):
        super(Encoder, self).__init__()
        self.m = enc
        self.dim_int = dim_in
        self.dim_out = dim_out

    def forward(self, x):
        return self.m(x).chunk(2, dim=1)


class Decoder(nn.Module):
    def __init__(self, dec, dim_in, dim_out):
        super(Decoder, self).__init__()
        self.m = dec
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self, z):
        return self.m(z)


class Transition(nn.Module):
    def __init__(self, trans, dim_z, dim_u):
        super(Transition, self).__init__()
        self.trans = trans
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.fc_B = nn.Linear(dim_z, dim_z * dim_u)
        self.fc_o = nn.Linear(dim_z, dim_z)

    def forward(self, h, Q, u):
        batch_size = h.size()[0]
        v, r = self.trans(h).chunk(2, dim=1)
        v1 = v.unsqueeze(2)
        rT = r.unsqueeze(1)
        I = Variable(torch.eye(self.dim_z).repeat(batch_size, 1, 1))
        if rT.data.is_cuda:
            I.dada.cuda()
        A = I.add(v1.bmm(rT))

        B = self.fc_B(h).view(-1, self.dim_z, self.dim_u)
        o = self.fc_o(h)

        # need to compute the parameters for distributions
        # as well as for the samples
        u = u.unsqueeze(2)

        d = A.bmm(Q.mu.unsqueeze(2)).add(B.bmm(u)).add(o).squeeze(2)
        sample = A.bmm(h.unsqueeze(2)).add(B.bmm(u)).add(o).squeeze(2)

        return sample, NormalDistribution(d, Q.sigma, Q.logsigma, v=v, r=r)

class RopeEncoder(Encoder):
    def __init__(self):
    # def __init__(self, dim_in=50*50, dim_out=80):
        m = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  
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
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        super(RopeEncoder, self).__init__(m)

class RopeDecoder(Decoder):
    def __init__(self):
    # def __init__(self, dim_in=80, dim_out=50*50):
        m = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(), 
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=2),
            nn.ReLU(),                                           
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=2),
            nn.ReLU(),                                         
            nn.ConvTranspose2d(32, 1, 2, stride=2, padding=2),
            nn.Sigmoid()
        )
        super(RopeDecoder, self).__init__(m)

class RopeTransition(Transition):
    def __init__(self, dim_z=80, dim_u=4):
        trans = nn.Sequential(
            nn.Linear(dim_z, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, dim_z*2)
        )
        super(RopeTransition, self).__init__(trans, dim_z, dim_u)

class PlaneEncoder(Encoder):
    def __init__(self, dim_in, dim_out):
        m = nn.Sequential(
            nn.Linear(dim_in, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Linear(150, dim_out*2)
        )
        super(PlaneEncoder, self).__init__(m, dim_in, dim_out)


class PlaneDecoder(Decoder):
    def __init__(self, dim_in, dim_out):
        m = nn.Sequential(
            nn.Linear(dim_in, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, dim_out),
            nn.BatchNorm1d(dim_out),
            nn.Sigmoid()
        )
        super(PlaneDecoder, self).__init__(m, dim_in, dim_out)


class PlaneTransition(Transition):
    def __init__(self, dim_z, dim_u):
        trans = nn.Sequential(
            nn.Linear(dim_z, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, dim_z*2)
        )
        super(PlaneTransition, self).__init__(trans, dim_z, dim_u)


class PendulumEncoder(Encoder):
    def __init__(self, dim_in, dim_out):
        m = nn.ModuleList([
            torch.nn.Linear(dim_in, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            torch.nn.Linear(800, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, 2 * dim_out)
        ])
        super(PendulumEncoder, self).__init__(m, dim_in, dim_out)

    def forward(self, x):
        for l in self.m:
            x = l(x)
        return x.chunk(2, dim=1)


class PendulumDecoder(Decoder):
    def __init__(self, dim_in, dim_out):
        m = nn.ModuleList([
            torch.nn.Linear(dim_in, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            torch.nn.Linear(800, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, dim_out),
            nn.Sigmoid()
        ])
        super(PendulumDecoder, self).__init__(m, dim_in, dim_out)

    def forward(self, z):
        for l in self.m:
            z = l(z)
        return z


class PendulumTransition(Transition):
    def __init__(self, dim_z, dim_u):
        trans = nn.Sequential(
            nn.Linear(dim_z, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, dim_z * 2),
            nn.BatchNorm1d(dim_z * 2),
            nn.Sigmoid() # Added to prevent nan
        )
        super(PendulumTransition, self).__init__(trans, dim_z, dim_u)


_CONFIG_MAP = {
    'plane': (PlaneEncoder, PlaneTransition, PlaneDecoder),
    'pendulum': (PendulumEncoder, PendulumTransition, PendulumDecoder),
    'rope': (RopeEncoder, RopeTransition, RopeDecoder)
}


def load_config(name):
    """Load a particular configuration
    Returns:
    (encoder, transition, decoder) A tuple containing class constructors
    """
    if name not in _CONFIG_MAP.keys():
        raise ValueError("Unknown config: %s", name)
    return _CONFIG_MAP[name]

# from deform.model.e2c import NormalDistribution

__all__ = ['load_config']
