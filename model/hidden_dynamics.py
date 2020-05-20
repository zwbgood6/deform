import numpy as np
import numpy
from numpy.linalg import inv, pinv, norm, det
import torch 



# def get_U(run_num, train_num):
#     '''assemble total_img_num actions in folder run_num 
#     run_num: string, e.g., 'run03'
#     total_img_num: number of images under 'run03'
#     '''
#     add1 = './rope_ddef get_control_matrix(G, U):
#     '''Calculate control matrix 
#     G: n * len(x_i)
#     U: n * len(u_i)
#     return control matrix L: len(x_i) * len(u_i) 
#     '''
#     n, d = np.shape(U)[0], np.shape(U)[1]
#     # U is a thin matrix
#     if n > d:
#         eps = 1e-5
#         return (pinv(U.T.dot(U)+eps*np.identity(d)).dot(U.T).dot(G)).T 
#     # U is a fat matrix   
#     elif n < d:
#         eps = 1e-5
#         return (U.T.dot(pinv(U.dot(U.T)+eps*np.identity(n))).dot(G)).T  
#     # U is a squared matrix       
#     else: 
#         return (inv(U).dot(G)).T

# def get_error(G, U, L):
#     '''||G-UL^T||^2
#     '''
#     return norm(G-U.dot(L.T), 2) ataset/rope/'
#     add2 = run_num
#     add3 = '/actions.npy'
#     U = np.load(add1+add2+add3)
#     return U[:train_num][:]

# def get_G(model, dataset):
#     n = dataset.__len__()
#     G = []
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     for idx in range(n):
#         data = dataset.__getitem__(idx)
#         data = data.float().to(device).view(-1, 1, 50, 50) # TODO: 50*50, input 250
#         latent = model.encoder(data).detach().cpu().numpy().reshape(-1).tolist()
#         G.append(latent)
#     return np.array(G[1:]) - np.array(G[:-1])


# def get_control_matrix(G, U):
#     '''Calculate control matrix 
#     G: n * len(x_i)
#     U: n * len(u_i)
#     return control matrix L: len(x_i) * len(u_i) 
#     '''
#     n, d = np.shape(U)[0], np.shape(U)[1]
#     # U is a thin matrix
#     if n > d:
#         eps = 1e-5
#         return (pinv(U.T.dot(U)+eps*np.identity(d)).dot(U.T).dot(G)).T 
#     # U is a fat matrix   
#     elif n < d:
#         eps = 1e-5
#         return (U.T.dot(pinv(U.dot(U.T)+eps*np.identity(n))).dot(G)).T  
#     # U is a squared matrix       
#     else: 
#         return (inv(U).dot(G)).T

# def get_error(G, U, L):
#     '''||G-UL^T||^2
#     '''
#     return norm(G-U.dot(L.T), 2) 

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
        return (torch.inverse(U.t().mm(U) + eps*torch.eye(d).mm(U.t()).mm(G))).t()
        #return (pinv(U.T.dot(U)+eps*np.identity(d)).dot(U.T).dot(G)).T 
    # U is a fat matrix   
    elif n < d:
        eps = 1e-5
        return (U.t().mm(torch.inverse(U.mm(U.t()) + eps*torch.eye(n))).mm(G)).t()  
        #return (U.T.dot(pinv(U.dot(U.T)+eps*np.identity(n))).dot(G)).T  
    # U is a squared matrix       
    else: 
        return (torch.inverse(U).mm(G)).t()
        # return (inv(U).dot(G)).T

def get_error(G, U, L):
    '''||G-UL^T||^2
    ''' 
    return torch.norm(G-U.mm(L.t()))

def get_next_state(embedded_state, action, L):
    '''get next embedded state after certain steps
    g_{t+k} = g_{t} + L * (u_{t} + u_{t+1} + ... + u_{t+k-1})

    embedded_state: 1 * len(x_i)
    action:         m * len(u_i), m is the number of predicted steps
    L:              len(x_i) * len(u_i), control matrix  
    '''
    sum_action = np.sum(action, axis=0)
    return embedded_state + sum_action.dot(L.T)

def get_step_error(embedded_state, action, L, ):
    '''get error with certain steps in latent space 

    '''
    
# if __name__ == "__main__":
#     U = get_U(run_num='run05') 