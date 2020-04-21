import gym
from gym import error, spaces
from gym.utils import closer, seeding
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from deform.utils.utils import *
from deform.utils.params import *
from gekko import GEKKO
from scipy.interpolate import CubicSpline

class RopeEnv(object):

    # metadata = {'render.modes': ['human', 'rgb_array'],
    #             'video.frames_per_second': 50}
    # reward_range = (-float('inf'), float('inf'))
    # spec = None

    def __init__(self, start_state, link_length, num_points, screen_width=4, screen_height=3):
        self.state = start_state
        self.link_length = link_length 
        self.num_points = num_points
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.x_init = start_state[0][0] 
        self.y_init = start_state[0][1]         
        self.max_position_x = 4
        self.min_position_x = 0
        self.max_position_y = 3
        self.min_position_y = 0
        #self.action_space = None
        #self.observation_space = None
        self.seed()
        self.viewer = None
        self.line_x_all = None 
        self.line_y_all = None
        self.new_line_x_all = None 
        self.new_line_y_all = None
        self.index = None
        self.move_angle = None
        self.move_length = None
        self.count = 0
        #self.m = GEKKO()

    def step(self, action):
        '''
        Observation:
            Type: n*2 array

        Action:    
            Type: Box(3)
            Num Action       Min  Max
            0   move_length  0    1
            1   move_angle   0    2*pi
            2   index        0    num_seg+1
        '''
        #raise NotImplementedError
        ob = self.state
        #link_length = self.link_length
        self.line_x_all, self.line_y_all = ob[:, 0], ob[:, 1]
        self.move_length, self.move_angle, self.index = action[0], action[1], action[2]
        self.index = int(self.index)
        self.new_line_x_all, self.new_line_y_all = generate_new_line(self.line_x_all, self.line_y_all, self.index, \
                                                            self.move_angle, self.move_length, self.link_length)
        self.state = next_ob = np.column_stack((self.new_line_x_all, self.new_line_y_all))                                                    
        done = False                                                                
        reward = -1
        info = None
        #self.line_x_all, self.line_y_all = self.new_line_x_all, self.new_line_y_all

        return next_ob, reward, done, info

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """
        x_all, y_all = generate_initial_points(self.x_init, self.y_init, self.num_points, self.link_length)
        self.count = 0

        raise np.transpose([x_all, y_all])

    def render(self, image_dir, spline=True, show_points=True, show_next_state=False, generate_video=False, video_dir=None, video_name=None):
        # generate frames
        if not generate_video:
            # screen
            plt.xlim(0, self.screen_width)
            plt.ylim(0, self.screen_height)

            if not spline:
                # current state
                plt.plot(self.line_x_all, self.line_y_all, c='b')
        
                # next state
                if show_next_state:
                    plt.plot(self.new_line_x_all, self.new_line_y_all, c='c')
            
            elif spline:
                # current state
                #x = np.linspace(self.line_x_all[0], self.line_x_all[self.num_points-1], num=self.num_points)   
                x = np.linspace(min(self.line_x_all), max(self.line_x_all), num=self.num_points)   
                y = np.c_[self.line_x_all, self.line_y_all]
                cs = CubicSpline(x, y, bc_type='natural')
                xs = np.linspace(min(self.line_x_all), max(self.line_x_all), 100)
                #plt.scatter(x_action_line , y_action_line, c='r', s=5)
                #plt.plot(self.line_x_all, self.line_y_all, c='b', label='data')
                plt.plot(cs(xs)[:, 0], cs(xs)[:, 1], 'b', label='cubic spline')

                # next state
                if show_next_state:
                    new_x = np.linspace(min(self.new_line_x_all), max(self.new_line_x_all), num=self.num_points)   
                    new_y = np.c_[self.new_line_x_all, self.new_line_y_all]
                    new_cs = CubicSpline(new_x, new_y, bc_type='natural')
                    new_xs = np.linspace(min(self.new_line_x_all), max(self.new_line_x_all), 100)
                    #plt.scatter(x_action_line , y_action_line, c='r', s=5)
                    #plt.plot(self.new_line_x_all, self.new_line_y_all, c='c', label='data')
                    plt.plot(new_cs(new_xs)[:, 0], new_cs(new_xs)[:, 1], 'c', label='cubic spline')           
            
            # points on a line
            if show_points:
                plt.scatter(self.line_x_all , self.line_y_all, c='r', s=5)
                if show_next_state:
                    plt.scatter(self.line_x_all , self.line_y_all, c='r', s=5)

            # action
            action = get_action(self.move_angle, self.move_length)
            grip_pos_before = np.array([self.line_x_all[self.index], self.line_y_all[self.index]])
            grip_pos_after = get_pos_after(grip_pos_before, action)
            x_action_line = [grip_pos_before[0], grip_pos_after[0]]
            y_action_line = [grip_pos_before[1], grip_pos_after[1]]
            plt.scatter(x_action_line , y_action_line, c='r', s=5)
            plt.arrow(grip_pos_before[0], grip_pos_before[1],\
                    action[0], action[1], width=0.01, length_includes_head=1, head_width=0.05) # plot.arrow(x,y,dx,dy): (x,y)-->(x+dx,y+dy)        

            # save figure         
            #dir = '/home/zwenbo/Documents/research/deform/deform/img/' + str(k) + '.png'
            plt.savefig(image_dir + str(self.count) + '.png')
            plt.close()
            
            # index increment  
            self.count += 1

        # generate video
        if generate_video:
            images_temp = [img for img in os.listdir(image_dir)] #if img.endswith(str(i) + ".png")]
            images = []
            for i in range(len(images_temp)):
                for j in images_temp:
                    directory = str(i) + '.png' 
                    if directory == j:
                        images.append(j)
            frame = cv2.imread(os.path.join(image_dir, images_temp[0]))
            height, width, _ = frame.shape
            video = cv2.VideoWriter(video_dir + video_name, 0, 1, (width,height))

            for image in images:
                video.write(cv2.imread(os.path.join(image_dir, image)))

            cv2.destroyAllWindows()
            video.release()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        # Sets the seed for this env's random number generator(s).
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

