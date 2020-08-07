import numpy as np

action_path = '/home/zwenbo/Documents/research/deform/rope_dataset/clean_dataset/simplified_clean_actions_all_size240.npy'
actions = np.load(action_path)
ratio = 50 / 240

for i in range(22515):
    action = actions[i]
    actions[i][:2] = action[:2] * ratio

new_action_path = './rope_dataset/clean_dataset/simplified_clean_actions_all_size50.npy'
np.save(new_action_path, actions)