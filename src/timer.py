import time
from settings import *

class Timer():

    def __init__(self) -> None:
        self.previous_time = time.time()
        self.camera_timer = 0
        self.dt = 0
        pass

    def update(self):
        now_time = time.time()
            # we have the delta time difference between frames
        self.dt = now_time - self.previous_time  # temporal distance between two succeding frames (more or less constant)
        self.dt *= TIME_SCALE
        self.previous_time = now_time
        self.camera_timer += self.dt 
        pass


