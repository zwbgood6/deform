import numpy as np                   
import math
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

def plot_grad_flow(named_parameters, folder_name):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    
    Source: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    '''
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)   
    plt.savefig('./result/{}/plot/gradient_flow.png'.format(folder_name))
    plt.close()

def create_loss_list(loss_logger, kld=True):
    if loss_logger is None:
        train_loss_all = []
        train_img_loss_all = []
        train_act_loss_all = []
        train_latent_loss_all = []
        train_pred_loss_all = []
        train_kld_loss_all = []
        test_loss_all = []
        test_img_loss_all = []
        test_act_loss_all = []
        test_latent_loss_all = []
        test_pred_loss_all = []            
        test_kld_loss_all = []
    else:
        train_loss_all = loss_logger['train_loss_all']
        train_img_loss_all = loss_logger['train_img_loss_all']
        train_act_loss_all = loss_logger['train_act_loss_all']
        train_latent_loss_all = loss_logger['train_latent_loss_all']
        train_pred_loss_all = loss_logger['train_pred_loss_all']       
        test_loss_all = loss_logger['test_loss_all']
        test_img_loss_all = loss_logger['test_img_loss_all']
        test_act_loss_all = loss_logger['test_act_loss_all']
        test_latent_loss_all = loss_logger['test_latent_loss_all']
        test_pred_loss_all = loss_logger['test_pred_loss_all']         
        if kld is True:
            train_kld_loss_all = loss_logger['train_kld_loss_all']
            test_kld_loss_all = loss_logger['test_kld_loss_all']
        else:
            train_kld_loss_all = []
            test_kld_loss_all = []                
    return train_loss_all, train_img_loss_all, train_act_loss_all, train_latent_loss_all, train_pred_loss_all, train_kld_loss_all, \
           test_loss_all, test_img_loss_all, test_act_loss_all, test_latent_loss_all, test_pred_loss_all, test_kld_loss_all    

def create_folder(folder_name):
    if not os.path.exists('./result/' + folder_name):
        os.makedirs('./result/' + folder_name)
    if not os.path.exists('./result/' + folder_name + '/plot'):
        os.makedirs('./result/' + folder_name + '/plot')
    if not os.path.exists('./result/' + folder_name + '/reconstruction_test'):
        os.makedirs('./result/' + folder_name + '/reconstruction_test')
    if not os.path.exists('./result/' + folder_name + '/reconstruction_train'):
        os.makedirs('./result/' + folder_name + '/reconstruction_train')
    if not os.path.exists('./result/' + folder_name + '/reconstruction_act_train'):
        os.makedirs('./result/' + folder_name + '/reconstruction_act_train')
    if not os.path.exists('./result/' + folder_name + '/reconstruction_act_test'):
        os.makedirs('./result/' + folder_name + '/reconstruction_act_test')    

def rect(poke, c, label=None):
    # from rope.ipynb in Berkeley's rope dataset file
    x, y, t, l = poke
    dx = -200 * l * math.cos(t)
    dy = -200 * l * math.sin(t)
    arrow = plt.arrow(x, y, dx, dy, width=0.001, head_width=6, head_length=6, color=c, label=label)        
    #plt.legend([arrow,], ['My label',])

def plot_action(resz_action, recon_action, directory):
    # from rope.ipynb in Berkeley's rope dataset file
    plt.figure()
    # upper row original
    plt.subplot(1, 2, 1)
    rect(resz_action[i], "blue")
    plt.axis('off') 
    # middle row reconstruction
    plt.subplot(1, 2, 2)
    rect(recon_action[i], "red")
    plt.axis('off')
    plt.savefig(directory) 
    plt.close()

def plot_sample(img_before, img_after, resz_action, recon_action, directory):
    # from rope.ipynb in Berkeley's rope dataset file
    plt.figure()
    N = int(img_before.shape[0])
    for i in range(N):
        # upper row original
        plt.subplot(3, N, i+1)
        rect(resz_action[i], "blue")
        plt.imshow(img_before[i].reshape((50,50)))
        plt.axis('off') 
        # middle row reconstruction
        plt.subplot(3, N, i+1+N)
        rect(recon_action[i], "red")
        plt.imshow(img_before[i].reshape((50,50)))
        plt.axis('off')
        # lower row: next image after action
        plt.subplot(3, N, i+1+2*N)
        plt.imshow(img_after[i].reshape((50,50)))
        plt.axis('off')
    plt.savefig(directory) 
    plt.close()

def plot_cem_sample(img_before, img_after, img_after_pred, resz_action, recon_action, directory):
    # from rope.ipynb in Berkeley's rope dataset file
    plt.figure()
    #N = int(img_before.shape[0])
    # upper row original
    plt.subplot(2, 2, 1)
    rect(resz_action, "blue", "Ground Truth Action")
    plt.imshow(img_before.reshape((50,50)), cmap='binary')
    plt.axis('off') 
    # middle row reconstruction
    plt.subplot(2, 2, 2)
    plt.imshow(img_after.reshape((50,50)), cmap='binary')
    plt.axis('off')
    # lower row: next image after action
    plt.subplot(2, 2, 3)
    rect(recon_action, "red", "Sampled Action")    
    plt.imshow(img_before.reshape((50,50)), cmap='binary')
    plt.axis('off')
    # lower row: next image after action
    plt.subplot(2, 2, 4)
    plt.imshow(img_after_pred.reshape((50,50)), cmap='binary')
    plt.axis('off')    
    plt.savefig(directory) 
    plt.close()

def plot_sample_multi_step(img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, act1, act2,\
    act3, act4, act5, act6, act7, act8, act9, act10, directory):
    # multi-step prediction with action on the image
    plt.figure()
    N = int(img1.shape[0])
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(N):
        # img1, act1
        plt.subplot(11, N, i+1)
        rect(act1[i], "red")
        plt.imshow(img1[i].reshape((50,50)), cmap='binary')
        plt.axis('off') 
        # img2, act2
        plt.subplot(11, N, i+1+N)
        rect(act2[i], "red")
        plt.imshow(img2[i].reshape((50,50)), cmap='binary')
        plt.axis('off')
        # img3, act3
        plt.subplot(11, N, i+1+2*N)
        rect(act3[i], "red")
        plt.imshow(img3[i].reshape((50,50)), cmap='binary')
        plt.axis('off')
        # img4, act4
        plt.subplot(11, N, i+1+3*N)
        rect(act4[i], "red")
        plt.imshow(img4[i].reshape((50,50)), cmap='binary')
        plt.axis('off')
        # img5, act5
        plt.subplot(11, N, i+1+4*N)
        rect(act5[i], "red")
        plt.imshow(img5[i].reshape((50,50)), cmap='binary')
        plt.axis('off')
        # img6, act6
        plt.subplot(11, N, i+1+5*N)
        rect(act6[i], "red")
        plt.imshow(img6[i].reshape((50,50)), cmap='binary')
        plt.axis('off')
        # img7, act7
        plt.subplot(11, N, i+1+6*N)
        rect(act7[i], "red")
        plt.imshow(img7[i].reshape((50,50)), cmap='binary')
        plt.axis('off')
        # img8, act8
        plt.subplot(11, N, i+1+7*N)
        rect(act8[i], "red")
        plt.imshow(img8[i].reshape((50,50)), cmap='binary')
        plt.axis('off')
        # img9, act9
        plt.subplot(11, N, i+1+8*N)
        rect(act9[i], "red")
        plt.imshow(img9[i].reshape((50,50)), cmap='binary')
        plt.axis('off')
        # img10, act10
        plt.subplot(11, N, i+1+9*N)
        rect(act10[i], "red")
        plt.imshow(img10[i].reshape((50,50)), cmap='binary')
        plt.axis('off')                                
        # img11, act11
        plt.subplot(11, N, i+1+10*N)
        plt.imshow(img11[i].reshape((50,50)), cmap='binary')
        plt.axis('off')    
    plt.savefig(directory) 
    plt.close()

def generate_initial_points(x, y, num_points, link_length):
    """generate initial points for a line
    x: the first (start from left) point's x position
    y: the first (start from left) point's y position
    num_points: number of points on a line
    link_length: each segment's length
    """
    x_all = [x]
    y_all = [y]
    for _ in range(num_points-1):
        phi = np.random.uniform(-np.pi/10, np.pi/10)
        #phi = np.random.uniform(0, np.pi/2)
        x1, y1 = x + link_length * np.cos(phi), y + link_length * np.sin(phi)
        x_all.append(x1)
        y_all.append(y1)
        x, y = x1, y1
    
    return x_all, y_all

def generate_new_line(line_x_all, line_y_all, index, move_angle, move_length, link_length):
    ###
    # line_x_all: x position for all points on a line
    #        i.e. np.array([x1, x2, x3,..., xn]) 
    # line_y_all: y position for all points on a line
    #        i.e. np.array([y1, y2, y3,..., yn])
    # index: touching point's index in the line. Scalar.
    # grip_pos_before: gripper's position before moving in 2D
    #                  assume it is the same as touching posiiton on a line
    #                  i.e. np.array([x, y])
    # grip_pos_after: gripper's position after moving in 2D
    # move_angle: gripper's moving angle. [0, 2*pi]
    # move_length: gripper's moving length. Scalar.
    # link_length: constant distance between two nearby points. Scalar.
    # num_points: number of points on the line. Scalar.
    # action: relative x and y position change.
    #         i.e. np.array([delta_x, delta_y])
    ###

    num_points = np.size(line_x_all)
    # initialize new_line_x_all and new_line_y_all
    new_line_x_all = [0] * num_points #np.zeros_like(line_x_all)
    new_line_y_all = [0] * num_points #np.zeros_like(line_y_all)

    # action
    action = get_action(move_angle, move_length)

    # gripper position (touching position on the line) before moving
    grip_pos_before = np.array([line_x_all[index], line_y_all[index]])

    # gripper position after moving
    grip_pos_after = get_pos_after(grip_pos_before, action)
    new_line_x_all[index] = grip_pos_after[0]
    new_line_y_all[index] = grip_pos_after[1]

    # move points in the left side in order
    if index != 0:        
        grip_pos_after_temp = grip_pos_after
        for i in range(index):
            new_index_left = index - (i+1)
            moved_pos_before = np.array([line_x_all[new_index_left], line_y_all[new_index_left]]) 
            moved_pos_after = generate_new_point_pos_on_the_line(grip_pos_after_temp, moved_pos_before, link_length)
            grip_pos_after_temp = moved_pos_after
            new_line_x_all[new_index_left] = grip_pos_after_temp[0]
            new_line_y_all[new_index_left] = grip_pos_after_temp[1]

    # move points in the right side in oder
    if index != (num_points-1):   
        grip_pos_after_temp = grip_pos_after     
        for j in range(num_points-index-1):
            new_index_right = index + (j+1)
            moved_pos_before = np.array([line_x_all[new_index_right], line_y_all[new_index_right]]) 
            moved_pos_after = generate_new_point_pos_on_the_line(grip_pos_after_temp, moved_pos_before, link_length)
            grip_pos_after_temp = moved_pos_after
            new_line_x_all[new_index_right] = grip_pos_after_temp[0]
            new_line_y_all[new_index_right] = grip_pos_after_temp[1]

    # # move points in the right side in order
    # # touch the line
    # line_index = get_line_index(gripper_x_pos, x_all)
    # touch_line_pos = [x_all[line_index], y_all[line_index], x_all[line_index+1], y_all[line_index+1]] # [x1, y1, x2, y2]
    
    # # move the touching line
    # touch_line_pos += [action[0], action[1], action[0], action[1]]

    # # 
    # #  
    return new_line_x_all, new_line_y_all

def generate_new_point_pos_on_the_line(grip_pos_after, moved_pos_before, link_length):
    # get relative positions between moved_pos_before and grip_pos_after 
    # cannot change the order of these two points
    delta_x = moved_pos_before[0] - grip_pos_after[0]
    delta_y = moved_pos_before[1] - grip_pos_after[1] 
    angle = math.atan2(delta_y, delta_x)

    # get nearby point position after moving
    x_after = grip_pos_after[0] + link_length * np.cos(angle)
    y_after = grip_pos_after[1] + link_length * np.sin(angle)
    moved_pos_after = np.array([x_after, y_after])
    
    return moved_pos_after

def get_action(angle, length):
    action = np.array([length*np.cos(angle), length*np.sin(angle)])
    
    return action

def get_pos_after(grip_pos_before, action):
    x, y = grip_pos_before[0], grip_pos_before[1]
    pos_after = np.array([x+action[0], y+action[1]])

    return pos_after    

def get_line_index(gripper_x_pos, x_all):
    line_index = sum(gripper_x_pos >= np.array(x_all)) - 1

    return line_index   

def collision_check():
    # check if gripper action moves the line   
    # move the line: return true
    # not move the line: return false

    return

def plot_train_loss(file_name, folder_name):
    train_loss = np.load(file_name)
    plt.figure()
    plt.plot(train_loss)
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./result/{}/plot/train_loss.png'.format(folder_name))
    plt.close()

def plot_test_loss(file_name, folder_name):
    test_loss = np.load(file_name)
    plt.figure()
    plt.plot(test_loss)
    plt.title('Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./result/{}/plot/test_loss.png'.format(folder_name))
    plt.close()    

def plot_train_img_loss(file_name, folder_name):
    img_loss = np.load(file_name)
    plt.figure()
    plt.plot(img_loss)
    plt.title('Train Image Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./result/{}/plot/train_image_loss.png'.format(folder_name))
    plt.close()

def plot_train_act_loss(file_name, folder_name):
    act_loss = np.load(file_name)
    plt.figure()
    plt.plot(act_loss)
    plt.title('Train Action Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./result/{}/plot/train_action_loss.png'.format(folder_name))
    plt.close()

def plot_train_latent_loss(file_name, folder_name):
    latent_loss = np.load(file_name)
    plt.figure()
    plt.plot(latent_loss)
    plt.title('Train Latent Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./result/{}/plot/train_latent_loss.png'.format(folder_name))
    plt.close()

def plot_train_pred_loss(file_name, folder_name):
    pred_loss = np.load(file_name)
    plt.figure()
    plt.plot(pred_loss)
    plt.title('Train Prediction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./result/{}/plot/train_prediction_loss.png'.format(folder_name))
    plt.close()

def plot_train_kld_loss(file_name, folder_name):
    kld_loss = np.load(file_name)
    plt.figure()
    plt.plot(kld_loss)
    plt.title('Train KL Divergence Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./result/{}/plot/train_kld_loss.png'.format(folder_name))
    plt.close()

def plot_test_img_loss(file_name, folder_name):
    img_loss = np.load(file_name)
    plt.figure()
    plt.plot(img_loss)
    plt.title('Test Image Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./result/{}/plot/test_image_loss.png'.format(folder_name))
    plt.close()

def plot_test_act_loss(file_name, folder_name):
    act_loss = np.load(file_name)
    plt.figure()
    plt.plot(act_loss)
    plt.title('Test Action Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./result/{}/plot/test_action_loss.png'.format(folder_name))
    plt.close()

def plot_test_latent_loss(file_name, folder_name):
    latent_loss = np.load(file_name)
    plt.figure()
    plt.plot(latent_loss)
    plt.title('Test Latent Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./result/{}/plot/test_latent_loss.png'.format(folder_name))
    plt.close()

def plot_test_pred_loss(file_name, folder_name):
    pred_loss = np.load(file_name)
    plt.figure()
    plt.plot(pred_loss)
    plt.title('Test Prediction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./result/{}/plot/test_prediction_loss.png'.format(folder_name))
    plt.close()

def plot_test_kld_loss(file_name, folder_name):
    kld_loss = np.load(file_name)
    plt.figure()
    plt.plot(kld_loss)
    plt.title('Test KL Divergence Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./result/{}/plot/test_kld_loss.png'.format(folder_name))
    plt.close()
# def plot_separate_loss(file_names, folder_name):
#     for file_name in file_names:


def plot_all_train_loss_with_noise(train, test, img, act, latent, pred, kld, folder_name):
    train_loss = np.load(train)#[10:]
    test_loss = np.load(test)#[10:]
    img_loss = np.load(img)#[10:]
    act_loss = np.load(act)#[10:]
    latent_loss = np.load(latent)#[10:]
    pred_loss = np.load(pred)#[10:]
    kld_loss = np.load(kld)#[10:]
    plt.figure()
    train_curve, = plt.plot(train_loss, label='Train')
    test_curve, = plt.plot(test_loss, label='Test')
    img_curve, = plt.plot(img_loss, label='Image')
    act_curve, = plt.plot(act_loss, label='Action')
    latent_curve, = plt.plot(latent_loss, label='Latent')
    pred_curve, = plt.plot(pred_loss, label='Prediction')
    kld_curve, = plt.plot(kld_loss, label='KL Divergence')
    plt.title('Train loss and its subcomponents')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend([train_curve, test_curve, img_curve, act_curve, latent_curve, pred_curve, kld_curve], ['Train', 'Test', 'Image', 'Action', 'Latent', 'Prediction', 'KL Divergence'])
    plt.savefig('./result/{}/plot/all_train_loss.png'.format(folder_name))
    plt.close()

def plot_all_test_loss_with_noise(test, img, act, latent, pred, kld, folder_name):
    test_loss = np.load(test)#[20:]
    img_loss = np.load(img)#[20:]
    act_loss = np.load(act)#[20:]
    latent_loss = np.load(latent)#[20:]
    pred_loss = np.load(pred)#[20:]
    kld_loss = np.load(kld)#[20:]
    plt.figure()
    test_curve, = plt.plot(test_loss, label='Test')
    img_curve, = plt.plot(img_loss, label='Image')
    act_curve, = plt.plot(act_loss, label='Action')
    latent_curve, = plt.plot(latent_loss, label='Latent')
    pred_curve, = plt.plot(pred_loss, label='Prediction')
    kld_curve, = plt.plot(kld_loss, label='KL Divergence')
    plt.title('Test loss and its subcomponents')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend([test_curve, img_curve, act_curve, latent_curve, pred_curve, kld_curve], ['Test', 'Image', 'Action', 'Latent', 'Prediction', 'KL Divergence'])
    plt.savefig('./result/{}/plot/all_test_loss.png'.format(folder_name))
    plt.close()

def plot_all_train_loss_without_noise(train, test, img, act, latent, pred, folder_name):
    train_loss = np.load(train)#[10:]
    test_loss = np.load(test)#[10:]
    img_loss = np.load(img)#[10:]
    act_loss = np.load(act)#[10:]
    latent_loss = np.load(latent)#[10:]
    pred_loss = np.load(pred)#[10:]
    plt.figure()
    train_curve, = plt.plot(train_loss, label='Train')
    test_curve, = plt.plot(test_loss, label='Test')
    img_curve, = plt.plot(img_loss, label='Image')
    act_curve, = plt.plot(act_loss, label='Action')
    latent_curve, = plt.plot(latent_loss, label='Latent')
    pred_curve, = plt.plot(pred_loss, label='Prediction')
    plt.title('Train loss and its subcomponents')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend([train_curve, test_curve, img_curve, act_curve, latent_curve, pred_curve], ['Train', 'Test', 'Image', 'Action', 'Latent', 'Prediction'])
    plt.savefig('./result/{}/plot/all_train_loss.png'.format(folder_name))
    plt.close()

def plot_all_test_loss_without_noise(test, img, act, latent, pred, folder_name):
    test_loss = np.load(test)#[20:]
    img_loss = np.load(img)#[20:]
    act_loss = np.load(act)#[20:]
    latent_loss = np.load(latent)#[20:]
    pred_loss = np.load(pred)#[20:]
    plt.figure()
    test_curve, = plt.plot(test_loss, label='Test')
    img_curve, = plt.plot(img_loss, label='Image')
    act_curve, = plt.plot(act_loss, label='Action')
    latent_curve, = plt.plot(latent_loss, label='Latent')
    pred_curve, = plt.plot(pred_loss, label='Prediction')
    plt.title('Test loss and its subcomponents')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend([test_curve, img_curve, act_curve, latent_curve, pred_curve], ['Test', 'Image', 'Action', 'Latent', 'Prediction'])
    plt.savefig('./result/{}/plot/all_test_loss.png'.format(folder_name))
    plt.close()

def save_data(folder_name, epochs, train_loss_all, train_img_loss_all, train_act_loss_all,
              train_latent_loss_all, train_pred_loss_all, 
              test_loss_all, test_img_loss_all, test_act_loss_all, test_latent_loss_all, 
              test_pred_loss_all, train_kld_loss_all=None, test_kld_loss_all=None, K=None, L=None):
    np.save('./result/{}/train_loss_epoch{}.npy'.format(folder_name, epochs), train_loss_all)
    np.save('./result/{}/train_img_loss_epoch{}.npy'.format(folder_name, epochs), train_img_loss_all)
    np.save('./result/{}/train_act_loss_epoch{}.npy'.format(folder_name, epochs), train_act_loss_all)
    np.save('./result/{}/train_latent_loss_epoch{}.npy'.format(folder_name, epochs), train_latent_loss_all)
    np.save('./result/{}/train_pred_loss_epoch{}.npy'.format(folder_name, epochs), train_pred_loss_all)
    np.save('./result/{}/test_loss_epoch{}.npy'.format(folder_name, epochs), test_loss_all)
    np.save('./result/{}/test_img_loss_epoch{}.npy'.format(folder_name, epochs), test_img_loss_all)
    np.save('./result/{}/test_act_loss_epoch{}.npy'.format(folder_name, epochs), test_act_loss_all)
    np.save('./result/{}/test_latent_loss_epoch{}.npy'.format(folder_name, epochs), test_latent_loss_all)
    np.save('./result/{}/test_pred_loss_epoch{}.npy'.format(folder_name, epochs), test_pred_loss_all)          
    if train_kld_loss_all is not None:
        np.save('./result/{}/train_kld_loss_epoch{}.npy'.format(folder_name, epochs), train_kld_loss_all)
    if test_kld_loss_all is not None:
        np.save('./result/{}/test_kld_loss_epoch{}.npy'.format(folder_name, epochs), test_kld_loss_all) 
    if K is not None:
        np.save('./result/{}/koopman_matrix.npy'.format(folder_name), K)   
    if L is not None:
        np.save('./result/{}/control_matrix.npy'.format(folder_name), L)                   


# epochs = 800
# folder_name = 'test_freeze_Kp_Lpa'
# noise = False # consider whether adding noise in the latent dynamics

# train = './result/{}/train_loss_epoch{}.npy'.format(folder_name, epochs)
# train_img = './result/{}/train_img_loss_epoch{}.npy'.format(folder_name, epochs)
# train_act = './result/{}/train_act_loss_epoch{}.npy'.format(folder_name, epochs)
# train_latent = './result/{}/train_latent_loss_epoch{}.npy'.format(folder_name, epochs)
# train_pred = './result/{}/train_pred_loss_epoch{}.npy'.format(folder_name, epochs)

# test = './result/{}/test_loss_epoch{}.npy'.format(folder_name, epochs)
# test_img = './result/{}/test_img_loss_epoch{}.npy'.format(folder_name, epochs)
# test_act = './result/{}/test_act_loss_epoch{}.npy'.format(folder_name, epochs)
# test_latent = './result/{}/test_latent_loss_epoch{}.npy'.format(folder_name, epochs)
# test_pred = './result/{}/test_pred_loss_epoch{}.npy'.format(folder_name, epochs)

# if noise:
#     train_kld = './result/{}/train_kld_loss_epoch{}.npy'.format(folder_name, epochs)
#     test_kld = './result/{}/test_kld_loss_epoch{}.npy'.format(folder_name, epochs)
#     plot_all_train_loss_with_noise(train, test, train_img, train_act, train_latent, train_pred, train_kld, folder_name)
#     plot_all_test_loss_with_noise(test, test_img, test_act, test_latent, test_pred, test_kld, folder_name)
# else:
#     plot_all_train_loss_without_noise(train, test, train_img, train_act, train_latent, train_pred, folder_name)
#     plot_all_test_loss_without_noise(test, test_img, test_act, test_latent, test_pred, folder_name)



