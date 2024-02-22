class Ball():

    def __init__(self, x=0, y=0) -> None:
        self.x = x
        self.y = y
        self.prev_est_vel = [0, 0]
        pass

    def set_pos(self, x, y):
        self.x = x
        self.y = y
        pass
