import numpy as np
from deform.model.model import *
from torch.nn import functional as F
import math

def sample_action(I, mean=None, cov=None):
    '''TODO: unit test
    each action sequence length: H
    number of action sequences: N
    '''
    action = [0]*4
    mutiplier = np.array([50, 50, 2*math.pi, 0.14])
    addition = np.array([0, 0, 0, 0.01])
    while I[action[0]][action[1]] == 0:
        if ((mean is None) and (cov is None)):
            action_base = np.random.uniform(low=0.0, high=1.0, size=4)
        else:
            action_base = np.random.multivariate_normal(mean, cov)
        action = np.multiply(action_base, mutiplier) + addition
    return action

def generate_next_pred_state(recon_model, dyn_model, img_pre, act_pre):
    # TODO: unit test
    '''generate next predicted state
    reconstruction model: recon_model
    dynamics model: dyn_model
    initial image: img_pre
    each action sequence length: H
    number of action sequences: N
    '''
    latent_img_pre, latent_act_pre, _, _, _ = recon_model(img_pre, act_pre, img_cur=None)
    K_T_pre, L_T_pre = dyn_model(img_pre, act_pre)
    recon_latent_img_cur = get_next_state_linear(latent_img_pre, latent_act_pre, K_T_pre, L_T_pre)
    return recon_model.decoder(recon_latent_img_cur)

def generate_next_pred_state_in_n_step(recon_model, dyn_model, img_initial, N, H, mean=None, cov=None):
    # TODO: unit test
    imgs = [None]*N
    actions = [None]*N
    for n in range(N):
        img_tmp = img_initial
        #action = [None]*H
        for h in range(H):
            action = sample_action(img_tmp, mean, cov)
            if h==0:
                actions[n] = action # only get the first action in each sequence
            img_tmp = generate_next_pred_state(recon_model, dyn_model, img_tmp, action)
        imgs[n] = img_tmp
    return imgs, actions

def loss_function_img(img_recon, img_goal, N):
    # TODO: unit test
    loss = [None]*N
    for n in range(N):
        loss[n] = F.binary_cross_entropy(img_recon[n].view(-1, 2500), img_goal.view(-1, 2500), reduction='sum')
    return loss

def main(recon_model, dyn_model, T, iteration, K, N, H, img_initial, img_goal):
    # TODO: unit test
    for t in range(T):
        if t==0:
            img_cur = img_initial
        #Initialize Q with uniform distribution 
        #action_base = np.random.uniform(low=0.0, high=1.0, size=4)
        for _ in range(iteration):
            #Sample N action sequences a{t:t+H-1} with length H from Q 
            #action_base = np.random.uniform(low=0.0, high=1.0, size=4)
            #Use model M to predict the next state using M action sequences 
            imgs_recon, sample_actions = generate_next_pred_state_in_n_step(recon_model, dyn_model, img_cur, N, H, mean, cov)
            #Calculate binary cross entropy loss for predicted image and goal image 
            loss = loss_function_img(imgs_recon, img_goal, N)
            #Select K action sequences with lowest loss 
            loss_index = np.argsort(loss, axis=0)
            sorted_sample_actions = sample_actions[loss_index]
            #Fit multivariate gaussian distribution to K samples 
            #(see how to fit algorithm: 
            #https://stackoverflow.com/questions/27230824/fit-multivariate-gaussian-distribution-to-a-given-dataset) 
            mean = np.mean(sorted_sample_actions[:K], axis=0)
            cov = np.cov(sorted_sample_actions[:K], rowvar=0)
    #end for

        #Execute action a{t}* with lowest loss 
        action_best = sorted_sample_actions[0]
        #Observe new image I{t+1} 
        img_cur = generate_next_pred_state(recon_model, dyn_model, img_cur, action_best)
        
    #end for

# time step to execute the action
T = 10
# interation for CEM
iteration = 2
# total number of samples for action sequences
N = 10
# K samples to fit multivariate gaussian distribution
K = 3
# length of action sequence
H = T
# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
recon_model = CAE().to(device)
dyn_model = SysDynamics().to(device)
img_initial = None
img_goal = None
main(recon_model, dyn_model, T, iteration, K, N, H, img_initial, img_goal)