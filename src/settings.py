import numpy as np

TIME_SCALE = 2.0
BOUNCE_TRESH = 50

# the kernel size for blurring the frame (to reduce noise), it must be odd
ODD_BLUR = 5
# hsv thresholds to filter out the players except the ball depending on the region it is found

# top
TOP_GREEN_LOWER = (20, 0, 150)
# middle
MIDDLE_GREEN_LOWER = (45, 35, 150)
# bottom
BOTTOM_GREEN_LOWER = (35, 55, 150)
# initial upper
GREEN_UPPER = (65, 255, 255)
# ROI
ROI = {
    "minX": 200,
    "maxX": 900,
    "minY": 250,
    "maxY": 1700
}
# OFFSET for the upper region
OFFSET = 100
# the number of points needed on the buffer to compute the differece between frames
BUFFER_POINTS = 2
# we need a mask for the area of the court, the points are based on the ROI
COURT_MASK_POINTS = np.array([(267, 141), (25, 681), (1422, 686), (1161, 144)])

SCALA_2D = 1.395

# Parameters for 2d projection
# Coefficients for polynomial interpolation of y coordinate
COEFF = {
    "a": 0.0000031904,
    "b": 0.00477667,
    "c": 3.419626,
    "d": 303.38164
}

# Fixed parameters for x coordinate computation
B15 = 314
SCALA_X = (401-128)/(1104-314)
D15 = 128
E13 = 400
FACTOR = 0.109079
B7 = 715




