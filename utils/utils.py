import numpy as np                   
import math
import matplotlib.pyplot as plt
import os

def create_loss_list(loss_logger):
    if loss_logger is None:
        train_loss_all = []
        train_img_loss_all = []
        train_act_loss_all = []
        train_latent_loss_all = []
        train_pred_loss_all = []
        test_loss_all = []
        test_img_loss_all = []
        test_act_loss_all = []
        test_latent_loss_all = []
        test_pred_loss_all = []
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
    return train_loss_all, train_img_loss_all, train_act_loss_all, train_latent_loss_all, train_pred_loss_all, \
           test_loss_all, test_img_loss_all, test_act_loss_all, test_latent_loss_all, test_pred_loss_all    

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

def rect(poke, c):
    # from rope.ipynb in Berkeley's rope dataset file
    x, y, t, l = poke
    dx = -200 * l * math.cos(t)
    dy = -200 * l * math.sin(t)
    plt.arrow(x, y, dx, dy, head_width=5, head_length=5, color=c, alpha=0.5)

def plot_sample(img_before, img_after, resz_action, recon_action, directory):
    # from rope.ipynb in Berkeley's rope dataset file
    plt.figure()
    N = int(img_before.shape[0])
    for i in range(N):
        # upper row original
        plt.subplot(2, N, i+1)
        rect(resz_action[i], "blue")
        plt.imshow(img_before[i].reshape((50,50)))
        plt.axis('off') 
        # lower row reconstruction
        plt.subplot(2, N, i+1+N)
        rect(recon_action[i], "red")
        plt.imshow(img_before[i].reshape((50,50)))
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
# def plot_separate_loss(file_names, folder_name):
#     for file_name in file_names:


def plot_all_train_loss(train, test, img, act, latent, pred, folder_name):
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
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend([train_curve, test_curve, img_curve, act_curve, latent_curve, pred_curve], ['Train', 'Test', 'Image', 'Action', 'Latent', 'Prediction'])
    plt.savefig('./result/{}/plot/all_train_loss.png'.format(folder_name))
    plt.close()

def plot_all_test_loss(test, img, act, latent, pred, folder_name):
    test_loss = np.load(test)#[10:]
    img_loss = np.load(img)#[10:]
    act_loss = np.load(act)#[10:]
    latent_loss = np.load(latent)#[10:]
    pred_loss = np.load(pred)#[10:]
    plt.figure()
    test_curve, = plt.plot(test_loss, label='Test')
    img_curve, = plt.plot(img_loss, label='Image')
    act_curve, = plt.plot(act_loss, label='Action')
    latent_curve, = plt.plot(latent_loss, label='Latent')
    pred_curve, = plt.plot(pred_loss, label='Prediction')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend([test_curve, img_curve, act_curve, latent_curve, pred_curve], ['Test', 'Image', 'Action', 'Latent', 'Prediction'])
    plt.savefig('./result/{}/plot/all_test_loss.png'.format(folder_name))
    plt.close()

def save_data(folder_name, epochs, train_loss_all, train_img_loss_all, train_act_loss_all,
              train_latent_loss_all, train_pred_loss_all, test_loss_all, test_img_loss_all,
              test_act_loss_all, test_latent_loss_all, test_pred_loss_all, L):
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
    np.save('./result/{}/control_matrix.npy'.format(folder_name), L)                   


# epochs = 30
# folder_name = 'test'
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

# plot_all_train_loss(train, test, train_img, train_act, train_latent, train_pred, folder_name)
# plot_all_test_loss(test, test_img, test_act, test_latent, test_pred, folder_name)

