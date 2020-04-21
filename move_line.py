import numpy as np                   
import math

def generate_new_line(line_x_all, line_y_all, index, move_angle, move_length, link_length, num_points):
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

    # initialize new_line_x_all and new_line_y_all
    new_line_x_all = np.zeros_like(line_x_all)
    new_line_y_all = np.zeros_like(line_y_all)

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