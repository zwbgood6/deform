import numpy as np

action = np.load("/home/zwenbo/Documents/research/deform/rope_dataset/clean_dataset/simplified_clean_actions_all_size50.npy")
action_new = np.concatenate((action[:20198], action[21192:]), axis=0)
np.save("/home/zwenbo/Documents/research/deform/rope_dataset/rope_clean_all/rope_clean_all_size50.npy", action_new)
