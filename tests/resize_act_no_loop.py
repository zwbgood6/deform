import numpy as np

action_path = './rope_dataset/simplified_dataset/rope_no_loop_all_gray/actions.npy'
actions = np.load(action_path)
ratio = 50 / 240

for i in range(22515):
    action = actions[i]
    actions[i][:2] = action[:2] * ratio

new_action_path = './rope_dataset/simplified_dataset/rope_no_loop_all_resize_gray/resize_actions.npy'
np.save(new_action_path, actions)