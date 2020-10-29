import numpy as np

action = np.load("/home/zwenbo/Documents/research/deform/plan_result/curve_KL/KL_action_1000.npy")
state = np.load("/home/zwenbo/Documents/research/deform/plan_result/curve_KL/KL_image_1000.npy")

print("action mean is:", action.mean())
print("action std is:", np.sqrt(action.std()))
print("state mean is:", state.mean())
print("state std is:", np.sqrt(state.std()))