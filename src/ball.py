class Ball():

    def __init__(self, x=0, y=0) -> None:
        self.x = x
        self.y = y
        self.prev_est_vel = [0, 0]
        self.x_2d = 0
        self.y_2d = 0
        self.bounce = False
        self.dx = 0
        self.dy = 0
        self.djoko = False
        pass

    def set_pos(self, x, y):
        self.x = x
        self.y = y
        pass

    def set_2d_pos(self, x_2d, y_2d):
        self.x_2d = x_2d
        self.y_2d = y_2d
        pass

    def set_dx(self, dx):
        self.dx = dx
