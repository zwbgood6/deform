import numpy as np
from matplotlib import pyplot as plt

def plot_train_loss(file_name, folder_name):
    train_loss = np.load(file_name)
    plt.figure()
    plt.plot(train_loss)
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./result/{}/plot/train_loss.png'.format(folder_name))
    plt.close()

folder_name = 'test_E2C_gpu'
loss = np.load("/home/zwenbo/Documents/research/deform/result/test_E2C_gpu_update_loss/test_bound_loss_epoch1000.npy")
print("loss is : ", loss)