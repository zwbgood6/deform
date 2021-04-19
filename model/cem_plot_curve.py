import numpy as np
from numpy.linalg import det
from numpy import sqrt
from deform.model.dl_model import *
from deform.model.create_dataset import *
from deform.model.hidden_dynamics import get_next_state_linear
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torch.distributions import Uniform
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
import torchvision.transforms.functional as TF
from PIL import Image
from deform.utils.utils import plot_cem_sample
import os

def sample_action(I, mean=None, cov=None):
    '''TODO: unit test
    each action sequence length: H
    number of action sequences: N
    '''
    action = torch.tensor([0]*4, dtype=torch.float) 
    multiplier = torch.tensor([50, 50, 2*math.pi, 0.14])
    addition = torch.tensor([0, 0, 0, 0.01])
    thres = 0.9
    if I[0][0][0][0] == 1.:
        if ((mean is None) and (cov is None)):
            action_base = Uniform(low=0.0, high=1.0).sample((4,))
            action = torch.mul(action_base, multiplier) + addition
        else:
            cov = add_eye(cov)
            action = MultivariateNormal(mean, cov).sample()
        action[0], action[1] = 0, 0
        return action

    while I[0][0][torch.floor(action[0]).type(torch.LongTensor)][torch.floor(action[1]).type(torch.LongTensor)] != 1.:
        if ((mean is None) and (cov is None)):
            action_base = Uniform(low=0.0, high=1.0).sample((4,))
            action = torch.mul(action_base, multiplier) + addition
        else:
            cov = add_eye(cov)
            action = MultivariateNormal(mean, cov).sample() 
            while torch.floor(action[0]).type(torch.LongTensor) >= 50 or torch.floor(action[1]).type(torch.LongTensor) >= 50:
                cov = add_eye(cov)
                action = MultivariateNormal(mean, cov).sample()

    return action

def generate_next_pred_state(recon_model, dyn_model, img_pre, act_pre):
    '''generate next predicted state
    reconstruction model: recon_model
    dynamics model: dyn_model
    initial image: img_pre
    each action sequence length: H
    number of action sequences: N
    '''
    latent_img_pre, latent_act_pre, _, _, _ = recon_model(img_pre.reshape((-1, 1, 50, 50)), act_pre.reshape((-1, 4)).type(torch.float), None)
    K_T_pre, L_T_pre = dyn_model(img_pre.reshape((-1, 1, 50, 50)), act_pre.reshape((-1, 4)).type(torch.float))
    recon_latent_img_cur = get_next_state_linear(latent_img_pre, latent_act_pre, K_T_pre, L_T_pre)
    return recon_model.decoder(recon_latent_img_cur)

def generate_next_pred_state_in_n_step(recon_model, dyn_model, img_initial, N, H, mean=None, cov=None):
    imgs = [None]*N
    actions = torch.tensor([[0.]*4]*N) 
    for n in range(N):
        img_tmp = img_initial
        for h in range(H):
            action = sample_action(img_tmp, mean, cov)
            if h==0:
                actions[n] = action 
            img_tmp = generate_next_pred_state(recon_model, dyn_model, img_tmp, action)
        imgs[n] = img_tmp
    return imgs, actions

def loss_function_img(img_recon, img_goal, N):
    loss = torch.as_tensor([0.]*N)
    for n in range(N):
        loss[n] = F.binary_cross_entropy(img_recon[n].view(-1, 2500), img_goal.view(-1, 2500), reduction='sum')
    return loss

def add_eye(cov):
    if det(cov)==0:
        return cov + torch.eye(4) * 0.000001
    else:
        return cov

def mahalanobis(dist, cov):
    '''dist = mu1 - mu2, mu1 & mu2 are means of two multivariate gaussian distribution
    matrix multiplication: dist^T * cov^(-1) * dist
    '''
    return (dist.transpose(0,1).mm(cov.inverse())).mm(dist)

def bhattacharyya(dist, cov1, cov2):
    '''source: https://en.wikipedia.org/wiki/Bhattacharyya_distance
    '''
    cov = (cov1 + cov2) / 2
    d1 = mahalanobis(dist.reshape((4,-1)), cov) / 8
    if det(cov)==0 or det(cov1)==0 or det(cov2)==0:
        return inf
    d2 = np.log(det(cov) / sqrt(det(cov1) * det(cov2))) / 2
    return d1 + d2

def main(recon_model, dyn_model, T, K, N, H, img_initial, img_goal, resz_act, step_i, KL):
    for t in range(T):
        print("***** Start Step {}".format(t))
        if t==0:
            img_cur = img_initial
        #Initialize Q with uniform distribution 
        mean = None
        cov = None
        mean_tmp = None
        cov_tmp = None
        converge = False   
        iter_count = 0     
        while not converge:
            imgs_recon, sample_actions = generate_next_pred_state_in_n_step(recon_model, dyn_model, img_cur, N, H, mean, cov)
            #Calculate binary cross entropy loss for predicted image and goal image 
            loss = loss_function_img(imgs_recon, img_goal, N)
            #Select K action sequences with lowest loss 
            loss_index = torch.argsort(loss)
            sorted_sample_actions = sample_actions[loss_index]
            #Fit multivariate gaussian distribution to K samples 
            #(see how to fit algorithm: 
            #https://stackoverflow.com/questions/27230824/fit-multivariate-gaussian-distribution-to-a-given-dataset) 
            mean = torch.mean(sorted_sample_actions[:K], dim=0).type(torch.DoubleTensor)
            cov = torch.from_numpy(np.cov(sorted_sample_actions[:K], rowvar=0)).type(torch.DoubleTensor)
            # iteration is based on convergence of Q
            if det(cov) == 0 or cov_tmp == None:
                mean_tmp = mean
                cov_tmp = cov
                continue
            else:
                if det(cov_tmp)==0:
                    mean_tmp = mean
                    cov_tmp = cov 
                    continue   
                else:            
                    p = MultivariateNormal(mean, cov)
                    q = MultivariateNormal(mean_tmp, cov_tmp)
                if kl_divergence(p, q) < KL: 
                    converge = True
                mean_tmp = mean
                cov_tmp = cov    
            
            print("***** At action time step {}, iteration {} *****".format(t, iter_count))
            iter_count += 1    

        #Execute action a{t}* with lowest loss 
        action_best = sorted_sample_actions[0] 
        action_loss = ((action_best.detach().cpu().numpy()-resz_act[:4])**2).mean(axis=None)
        #Observe new image I{t+1} 
        img_cur = generate_next_pred_state(recon_model, dyn_model, img_cur, action_best)
        img_loss = F.binary_cross_entropy(img_cur.view(-1, 2500), img_goal.view(-1, 2500), reduction='mean')
        print("***** Generate Next Predicted Image {}*****".format(t+1))

    print("***** End Planning *****")
    return action_loss, img_loss.detach().cpu().numpy()

# plan result folder name
plan_folder_name = 'curve_KL'
if not os.path.exists('./plan_result/{}'.format(plan_folder_name)):
    os.makedirs('./plan_result/{}'.format(plan_folder_name))
# time step to execute the action
T = 1
# total number of samples for action sequences
N = 100
# K samples to fit multivariate gaussian distribution (N>K, K>1)
K = 50 
# length of action sequence
H = 1
# model
torch.manual_seed(1)
device = torch.device("cpu")
print("Device is:", device)
recon_model = CAE().to(device)
dyn_model = SysDynamics().to(device)

# action
# load GT action
resz_act_path = './rope_dataset/rope_no_loop_all_resize_gray_clean/simplified_clean_actions_all_size50.npy'
resz_act = np.load(resz_act_path)

# checkpoint
print('***** Load Checkpoint *****')
folder_name = "test_act80_pred30"
PATH = './result/{}/checkpoint'.format(folder_name)
checkpoint = torch.load(PATH, map_location=device)
recon_model.load_state_dict(checkpoint['recon_model_state_dict'])  
dyn_model.load_state_dict(checkpoint['dyn_model_state_dict'])  

total_img_num = 22515
image_paths_bi = create_image_path('rope_no_loop_all_resize_gray_clean', total_img_num)


def get_image(i):
    img = TF.to_tensor(Image.open(image_paths_bi[i])) > 0.3
    return img.reshape((-1, 1, 50, 50)).type(torch.float)


for KL in [1000]:
    action_loss_all = []
    img_loss_all = []
    for i in range(20000, 20010):
        img_initial = get_image(i)
        img_goal = get_image(i+1)
        action_loss, img_loss = main(recon_model, dyn_model, T, K, N, H, img_initial, img_goal, resz_act[i], i, KL)
        action_loss_all.append(action_loss)
        img_loss_all.append(img_loss)
    np.save('./plan_result/{}/KL_action_{}.npy'.format(plan_folder_name, KL), action_loss_all)
    np.save('./plan_result/{}/KL_image_{}.npy'.format(plan_folder_name, KL), img_loss_all)