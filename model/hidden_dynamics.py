import numpy as np
import numpy
from numpy.linalg import inv, pinv, norm, det
import torch 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_control_matrix(G, U):
    '''Calculate control matrix 
    G: n * len(x_i)
    U: n * len(u_i)
    return control matrix L: len(x_i) * len(u_i) 
    '''
    n, d = U.size()
    # U is a thin matrix
    if n > d:
        eps = 1e-5
        return (torch.pinverse(U.t().mm(U) + eps*torch.eye(d).to(device).mm(U.t()).mm(G)).to(device)).t()
    # U is a fat matrix   
    elif n < d:
        eps = 1e-5
        return (U.t().mm(torch.pinverse(U.mm(U.t()) + eps*torch.eye(n).to(device))).to(device).mm(G)).t()  
    # U is a squared matrix       
    else:
        return (torch.inverse(U).to(device).mm(G)).t()


def get_error(G, U, L):
    '''||G-UL^T||^2
    ''' 
    return torch.norm(G-U.mm(L.t()))

def get_error_linear(G, U, L_T):
    '''||G-UL_T||^2
    '''     
    err = G - torch.matmul(U.view(U.shape[0], 1, -1), L_T)
    return torch.norm(err.view(err.shape[0], -1))

def get_next_state(latent_image_pre, latent_action, K, L):
    '''get next embedded state after certain steps
    g_{t+1} = K * g_{t} + L * u_{t}

    embedded_state: 1 * len(x_i)
    action:         m * len(u_i), m is the number of predicted steps
    L:              len(x_i) * len(u_i), control matrix  
    '''

    return latent_image_pre.mm(K.t().to(device)) + latent_action.mm(L.t().to(device))

def get_next_state_linear(latent_image_pre, latent_action, K_T, L_T, z=None):
    '''
    latent_image_pre: (batch_size, latent_state_dim)
    latent_action: (batch_size, latent_act_dim)
    K_T: (batch_size, latent_state_dim, latent_state_dim)
    L_T: (batch_size, latent_act_dim, latent_state_dim)
    '''
    if z is not None:
        return (torch.matmul(latent_image_pre.view(latent_image_pre.shape[0], 1, -1), K_T)).view(latent_image_pre.shape[0], -1) + \
            (torch.matmul(latent_action.view(latent_action.shape[0], 1, -1), L_T)).view(latent_action.shape[0], -1) + z.to(device)
    else:
        return (torch.matmul(latent_image_pre.view(latent_image_pre.shape[0], 1, -1), K_T)).view(latent_image_pre.shape[0], -1) + \
            (torch.matmul(latent_action.view(latent_action.shape[0], 1, -1), L_T)).view(latent_action.shape[0], -1)         

def get_next_state_linear_without_L(latent_image_pre, K_T, z=None):
    '''
    latent_image_pre: (batch_size, latent_state_dim)
    latent_action: (batch_size, latent_act_dim)
    K_T: (batch_size, latent_state_dim, latent_state_dim)
    '''
    if z is not None:
        return (torch.matmul(latent_image_pre.view(latent_image_pre.shape[0], 1, -1), K_T)).view(latent_image_pre.shape[0], -1) + z.to(device)
    else:
        return (torch.matmul(latent_image_pre.view(latent_image_pre.shape[0], 1, -1), K_T)).view(latent_image_pre.shape[0], -1) 
             
