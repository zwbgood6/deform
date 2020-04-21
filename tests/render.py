def render(self, mode='human'):
    """Renders the environment.
    The set of supported modes varies per environment. (And some
    environments do not support rendering at all.) By convention,
    if mode is:
    - human: render to the current display or terminal and
        return nothing. Usually for human consumption.
    - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
        representing RGB values for an x-by-y pixel image, suitable
        for turning into a video.
    - ansi: Return a string (str) or StringIO.StringIO containing a
        terminal-style text representation. The text can include newlines
        and ANSI escape sequences (e.g. for colors).
    Note:
        Make sure that your class's metadata 'render.modes' key includes
            the list of supported modes. It's recommended to call super()
            in implementations to use the functionality of this method.
    Args:
        mode (str): the mode to render with
    Example:
    class MyEnv(Env):
        metadata = {'render.modes': ['human', 'rgb_array']}
        def render(self, mode='human'):
            if mode == 'rgb_array':
                return np.array(...) # return RGB frame suitable for video
            elif mode == 'human':
                ... # pop up a window and render
            else:
                super(MyEnv, self).render(mode=mode) # just raise an exception
    """
    world_width = self.max_position_x - self.min_position_x
    world_height = self.max_position_y - self.min_position_y
    scale_x = screen_width / world_width
    scale_y = screen_height / world_height

    if self.line_x_all is None or self.line_y_all is None \
        or self.new_line_x_all is None or self.new_line_y_all is None:
        return None

    #lines = [0] * (self.num_points - 1)

    if self.viewer is None:
        from gym.envs.classic_control import rendering
        self.viewer = rendering.Viewer(screen_width, screen_height)
        points = [0] * self.num_points
        self.points_trans = [0] * self.num_points
        lines = [0] * (self.num_points - 1)
        self.lines_trans = [0] * (self.num_points - 1)

        for i in range(self.num_points):
            # draw point
            self.x1 = (self.line_x_all[i] - self.min_position_x) * scale_x
            self.y1 = (self.line_y_all[i] - self.min_position_y) * scale_y                
            points[i] = rendering.make_circle(radius=2)
            points[i].set_color(1., 0., 0.)
            self.points_trans[i] = rendering.Transform(translation=(self.x1, self.y1))
            points[i].add_attr(self.points_trans[i])
            self.viewer.add_geom(points[i])
            
            # draw line
            if i == self.num_points - 1:
                continue
            self.x2 = (self.line_x_all[i+1] - self.min_position_x) * scale_x
            self.y2 = (self.line_y_all[i+1] - self.min_position_y) * scale_y
            lines[i] = rendering.Line(start=(self.x1, self.y1), end=(self.x2, self.y2))
            lines[i].set_color(.8, 0, 0)
            self.lines_trans[i] = rendering.Transform()
            lines[i].add_attr(self.lines_trans[i])
            self.viewer.add_geom(lines[i])
            self.angle = np.arctan2(self.y2 - self.y1, self.x2 - self.x1)
            #self.angle = np.arctan2(y2, x2)
    # update the new line position
    for j in range(self.num_points):
        # update points
        #delta_x = (self.new_line_x_all[j] - self.line_x_all[j]) * scale_x
        #delta_y = (self.new_line_y_all[j] - self.line_y_all[j]) * scale_y
        #self.points_trans[j].set_translation(delta_x, delta_y)
        self.new_x1 = (self.new_line_x_all[j] - self.min_position_x) * scale_x
        self.new_y1 = (self.new_line_y_all[j] - self.min_position_y) * scale_y   
        self.points_trans[j].set_translation(self.new_x1, self.new_y1)

        # update lines
        if j == self.num_points - 1:
            continue
        # step 1: rotate the line
        self.new_x2 = (self.new_line_x_all[j+1] - self.min_position_x) * scale_x
        self.new_y2 = (self.new_line_y_all[j+1] - self.min_position_y) * scale_y 
        angle = np.arctan2(self.new_y2 - self.new_y1, self.new_x2 - self.new_x1)           
        delta_angle = angle - self.angle
        self.lines_trans[j].set_rotation(delta_angle)
        # step 2: translate the line based on temporary (x,y) and desired (x,y) 
        temp_x1 = np.cos(delta_angle) * self.x1 - np.sin(delta_angle) * self.y1
        temp_y1 = np.sin(delta_angle) * self.x1 + np.cos(delta_angle) * self.y1
        delta_x = self.new_x1 - temp_x1
        delta_y = self.new_y1 - temp_y1
        self.lines_trans[j].set_translation(delta_x, delta_y)
        #self.lines_trans[j].set_translation(self.new_x1, self.new_y1)
        self.x1, self.y1, self.x2, self.y2 = self.new_x1, self.new_y1, self.new_x2, self.new_y2
        self.angle = angle
        #self.lines_trans[j].set_translation(x1, y1)
        #self.lines_trans[j].set_translation(delta_x, delta_y)
        #angle = np.arctan2(y2 - y1, x2 - x1)
        #delta_angle = angle - self.angle
        #self.lines_trans[j].set_translation(self.new_x1, self.new_y1)
        #angle1 = np.arctan2(self.new_line_y_all[j+1]-self.new_line_y_all[j], self.new_line_x_all[j+1]-self.new_line_x_all[j])
        #angle2 = np.arctan2(self.line_y_all[j+1]-self.line_y_all[j], self.line_x_all[j+1]-self.line_x_all[j])
        #delta_angle = angle2 - angle1
        #self.lines_trans[j].set_rotation(delta_angle)
    #delta_y = self.new_line_y_all[i] - self.line_y_all[i]
    #self.points_trans[i].set_translation(delta_x, delta_y)

    return self.viewer.render(return_rgb_array = mode=='rgb_array')