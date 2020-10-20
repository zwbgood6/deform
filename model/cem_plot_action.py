import torch
import numpy as np
import matplotlib.pyplot as plt
import math

def rect(poke, c):
    # from rope.ipynb in Berkeley's rope dataset file
    x, y, t, l = poke#[0], poke[1], poke[2], poke[3]
    dx = -2000 * l * math.cos(t)
    dy = -2000 * l * math.sin(t)
    plt.arrow(x, y, dx, dy, width=0.01, head_width=1, head_length=3, color=c)

def plot_sample(img_before, img_after, img_after_pred, resz_action, recon_action, directory):
    # from rope.ipynb in Berkeley's rope dataset file
    plt.figure()
    N = int(img_before.shape[0])
    # upper row original
    plt.subplot(2, 2, 1)
    rect(resz_action[i], "blue")
    plt.imshow(img_before.reshape((50,50)))
    plt.axis('off') 
    # middle row reconstruction
    plt.subplot(2, 2, 2)
    plt.imshow(img_after.reshape((50,50)))
    plt.axis('off')
    # lower row: next image after action
    plt.subplot(2, 2, 3)
    rect(recon_action[i], "blue")    
    plt.imshow(img_before.reshape((50,50)))
    plt.axis('off')
    # lower row: next image after action
    plt.subplot(2, 2, 4)
    plt.imshow(img_after_pred.reshape((50,50)))
    plt.axis('off')    
    plt.savefig(directory) 
    plt.close()

def plot_action(resz_action, recon_action, directory):
    # from rope.ipynb in Berkeley's rope dataset file
    plt.figure()
    # upper row original
    plt.subplot(1, 2, 1)
    rect(resz_action, "blue")
    plt.axis('off') 
    # middle row reconstruction
    plt.subplot(1, 2, 2)
    rect(recon_action, "red")
    plt.axis('off')
    plt.savefig(directory) 
    plt.close()


def get_image(i):
    img = TF.to_tensor(Image.open(image_paths_bi[i])) > 0.3
    return img.reshape((-1, 1, 50, 50)).type(torch.float)

# total number of samples for action sequences
N = 500
# K samples to fit multivariate gaussian distribution (N>K, K>1)
K = 50

# load GT action
resz_act_path = './rope_dataset/rope_no_loop_all_resize_gray_clean/simplified_clean_actions_all_size50.npy'
resz_act = np.load(resz_act_path)

# plot_sample(batch_data['image_bi_cur'][:n].detach().cpu().numpy(), 
#             batch_data['image_bi_post'][:n].detach().cpu().numpy(), 
#             batch_data['resz_action_cur'][:n].detach().cpu().numpy(), 
#             recon_act.view(-1, 4)[:n].detach().cpu().numpy(), 
#                 './result/{}/reconstruction_act_test/recon_epoch_{}.png'.format(folder_name, epoch))  

# load image
total_img_num = 22515
image_paths_bi = create_image_path('rope_no_loop_all_resize_gray_clean', total_img_num)

for i in range(45):
    img_before = get_image(i)
    img_after = get_image(i+1)    
    action_best = torch.load("./plan_result/03/action_best_step{}_N{}_K{}.pt".format(i, N, K))
    # plot_action(resz_act[i][:4], action_best.cpu().numpy(), 
    #             './plan_result/03/compare_action_{}'.format(i))
    plot_sample(img_before, 
                img_after, 
                img_after_pred, 
                resz_action, 
                recon_action, 
                './plan_result/03/compare_align_{}'.format(i))