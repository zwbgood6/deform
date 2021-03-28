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
    #action = torch.IntTensor([0]*4) 
    action = torch.tensor([0]*4, dtype=torch.float) 
    multiplier = torch.tensor([50, 50, 2*math.pi, 0.14])
    addition = torch.tensor([0, 0, 0, 0.01])
    thres = 0.9
    # consider edge case when I[0][0]!=0 in the beginning
    if I[0][0][0][0] == 1.: # > thres means grasping white rope, first two zeros are unrelated dimensional index
        if ((mean is None) and (cov is None)):
            #action_base = np.random.uniform(low=0.0, high=1.0, size=4)
            action_base = Uniform(low=0.0, high=1.0).sample((4,))
            action = torch.mul(action_base, multiplier) + addition
        else:
            cov = add_eye(cov)
            #action_base = np.random.multivariate_normal(mean, cov)
            action = MultivariateNormal(mean, cov).sample()
        #action = np.multiply(action_base, mutiplier) + addition        
        #action = torch.mul(action_base, multiplier) + addition
        action[0], action[1] = 0, 0
        return action
    # find grasping point TODO: there is a problem here, CHECK!!
    while I[0][0][torch.floor(action[0]).type(torch.LongTensor)][torch.floor(action[1]).type(torch.LongTensor)] != 1.: # < thres means grasping black region, 255 is max pixtel intensity for white color 
        if ((mean is None) and (cov is None)):
            #action_base = np.random.uniform(low=0.0, high=1.0, size=4)
            action_base = Uniform(low=0.0, high=1.0).sample((4,))
            action = torch.mul(action_base, multiplier) + addition
        else:
            # det(cov) cannot be zero, add small trace value to cov TODO: see if this can work?
            #cov = add_eye(cov)
            cov = add_eye(cov)
            action = MultivariateNormal(mean, cov).sample() # dont need to use multiplication and addition
            while torch.floor(action[0]).type(torch.LongTensor) >= 50 or torch.floor(action[1]).type(torch.LongTensor) >= 50:
                cov = add_eye(cov)
                action = MultivariateNormal(mean, cov).sample()
        #action = torch.mul(action_base, multiplier) + addition            
        #action = torch.mul(action_base, multiplier.cuda()) + addition.cuda()     
        #action = np.multiply(action_base, mutiplier) + addition
    return action

def generate_next_pred_state(e2c_model, img_pre, act_pre):
    # TODO: unit test
    '''generate next predicted state
    reconstruction model: recon_model
    dynamics model: dyn_model
    initial image: img_pre
    each action sequence length: H
    number of action sequences: N
    '''
    recon_img_cur = e2c_model.predict(img_pre, act_pre)
    return recon_img_cur
    # latent_img_pre, latent_act_pre, _, _, _ = recon_model(img_pre.reshape((-1, 1, 50, 50)), act_pre.reshape((-1, 4)).type(torch.float), None)
    # K_T_pre, L_T_pre = dyn_model(img_pre.reshape((-1, 1, 50, 50)), act_pre.reshape((-1, 4)).type(torch.float))
    # recon_latent_img_cur = get_next_state_linear(latent_img_pre, latent_act_pre, K_T_pre, L_T_pre)
    # return recon_model.decoder(recon_latent_img_cur)

def generate_next_pred_state_in_n_step(e2c_model, img_initial, N, H, mean=None, cov=None):
    # TODO: unit test
    imgs = [None]*N
    actions = torch.tensor([[0.]*4]*N) #torch.tensor(np.nan_to_num(np.array([None]*N)))
    for n in range(N):
        img_tmp = img_initial
        #action = [None]*H
        for h in range(H):
            action = sample_action(img_tmp, mean, cov)
            if h==0:
                actions[n] = action # only get the first action in each sequence
            img_tmp = generate_next_pred_state(e2c_model, img_tmp, action)
        imgs[n] = img_tmp
    return imgs, actions

def loss_function_img(img_recon, img_goal, N):
    # TODO: unit test
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
    # if cov1==None or cov2==None:
    #     return inf
    cov = (cov1 + cov2) / 2
    d1 = mahalanobis(dist.reshape((4,-1)), cov) / 8
    if det(cov)==0 or det(cov1)==0 or det(cov2)==0:
        return inf
    #cov, cov1, cov2 = add_eye(cov), add_eye(cov1), add_eye(cov2) 
    d2 = np.log(det(cov) / sqrt(det(cov1) * det(cov2))) / 2
    return d1 + d2

def main(e2c_model, T, K, N, H, img_initial, img_goal, resz_act, step_i, KL):
    # TODO: unit test
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
        #for _ in range(iteration):
            #Sample N action sequences a{t:t+H-1} with length H from Q 
            #action_base = np.random.uniform(low=0.0, high=1.0, size=4)
            #Use model M to predict the next state using M action sequences 
            imgs_recon, sample_actions = generate_next_pred_state_in_n_step(e2c_model, img_cur, N, H, mean, cov)
            #Calculate binary cross entropy loss for predicted image and goal image 
            loss = loss_function_img(imgs_recon, img_goal, N)
            #Select K action sequences with lowest loss 
            #loss_sort, loss_index = torch.sort(loss)
            loss_index = torch.argsort(loss)
            # sorted_sample_actions = torch.as_tensor(sample_actions[loss_index])
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
                if kl_divergence(p, q) < KL: # 0.3 is okay
                    converge = True
                # if bhattacharyya(mean_tmp-mean, cov_tmp, cov) < 0.2: # tune 0.2
                #     converge = True
                mean_tmp = mean
                cov_tmp = cov    
            
            print("***** At action time step {}, iteration {} *****".format(t, iter_count))
            iter_count += 1    
    #end for

        #Execute action a{t}* with lowest loss 
        action_best = sorted_sample_actions[0]
        action_loss = ((action_best.detach().cpu().numpy()-resz_act[:4])**2).mean(axis=None)
        #Observe new image I{t+1} 
        img_cur = generate_next_pred_state(e2c_model, img_cur, action_best)
        img_loss = F.binary_cross_entropy(img_cur.view(-1, 2500), img_goal.view(-1, 2500), reduction='mean')
        #comparison = torch.cat([img_initial, img_goal, img_cur])
        #save_image(comparison, "./plan_result/{}/image_best_step{}_N{}_K{}.png".format(plan_folder_name, step_i, N, K))
        # plot_cem_sample(img_initial.detach().reshape((50,50)).cpu().numpy(), 
        #                 img_goal.detach().reshape((50,50)).cpu().numpy(), 
        #                 img_cur.detach().reshape((50,50)).cpu().numpy(), 
        #                 resz_act[:4],  
        #                 action_best.detach().cpu().numpy(), 
        #                 './plan_result/{}/compare_align_{}.png'.format(plan_folder_name, i))        
        print("***** Generate Next Predicted Image {}*****".format(t+1))
    #end for
    #comparison_gt = torch.cat([img_initial, img_goal, img_cur])
    print("***** End Planning *****")
    return action_loss, img_loss.detach().cpu().numpy()

# plan result folder name
plan_folder_name = 'test_e2c'
if not os.path.exists('./plan_result/{}'.format(plan_folder_name)):
    os.makedirs('./plan_result/{}'.format(plan_folder_name))
# time step to execute the action
T = 1 # TODO: I cannot only check the correntness of T=1 now
# iteration for CEM
#iteration = 2
# total number of samples for action sequences
N = 100
# K samples to fit multivariate gaussian distribution (N>K, K>1)
K = 50 # 50 TODO: every action has its own distribution
# length of action sequence
H = 1 # 10-50
# model
torch.manual_seed(1)
device = torch.device("cpu")
print("Device is:", device)
e2c_model = E2C().to(device)

# action
# load GT action
resz_act_path = './rope_dataset/rope_no_loop_all_resize_gray_clean/simplified_clean_actions_all_size50.npy'
resz_act = np.load(resz_act_path)

# checkpoint
print('***** Load Checkpoint *****')
folder_name = "test_E2C_gpu_update_loss"
PATH = './result/{}/checkpoint'.format(folder_name)
checkpoint_e2c = torch.load(PATH, map_location=device)
e2c_model.load_state_dict(checkpoint_e2c['e2c_model_state_dict'])  

total_img_num = 22515
#train_num = int(total_img_num * 0.8)
image_paths_bi = create_image_path('rope_no_loop_all_resize_gray_clean', total_img_num)


#resz_act_path = './rope_dataset/rope_no_loop_all_resize_gray_clean/simplified_clean_actions_all_size50.npy'
#resz_act = np.load(resz_act_path)

#dataset = MyDataset(image_paths_bi, resz_act, transform=ToTensor())

def get_image(i):
    img = TF.to_tensor(Image.open(image_paths_bi[i])) > 0.3
    return img.reshape((-1, 1, 50, 50)).type(torch.float)

for KL in [1000]:
    action_loss_all = []
    img_loss_all = []
    for i in range(20000, 20010):
        img_initial = get_image(i)
        img_goal = get_image(i+1)
        action_loss, img_loss = main(e2c_model, T, K, N, H, img_initial, img_goal, resz_act[i], i, KL)
        action_loss_all.append(action_loss)
        img_loss_all.append(img_loss)
    np.save('./plan_result/{}/KL_action_{}.npy'.format(plan_folder_name, KL), action_loss_all)
    np.save('./plan_result/{}/KL_image_{}.npy'.format(plan_folder_name, KL), img_loss_all)