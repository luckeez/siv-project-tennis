from collections import deque

class TennisField():

    def __init__(self, args) -> None:
        self.direction = ""
        self.insideCourt = ""
        self.isBallInUpperRegion = False
        self.greenLower = (30, 55, 150)
        self.args = args
        self.pts = deque(maxlen=self.args["buffer"])
        pass

