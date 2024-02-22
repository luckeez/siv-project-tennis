from collections import deque
import argparse
import cv2
import time
import numpy as np
import math
from timer import Timer

from settings import *
from utils import *

import csv

# arguments, need to provide the video link. eg. ~/main.py --video ~/Images/tennis_match_2.mp4
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

args["video"] = "tennis_match_2.mp4"

# init parameters
# points where the ball is detected
pts = deque(maxlen=args["buffer"])
# direction message the ball is heading towards
direction = ""
# inside or outside court message
insideCourt = ""
# variables used to check the bouncing of the ball
# camera_timer = 0
timer = Timer()
(dX, dY) = (0, 0)
prev_est_vel = [0, 0]
est_vel = [0, 0]
# we assume the ball starts at the bottom
isBallInUpperRegion = False

# substracts the background from the frame
backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=75, detectShadows=False)

# initial
greenLower = (30, 55, 150)

# we use the video reference from the arguments
vs = cv2.VideoCapture(args["video"])

# we get the video fps
camera_fps = vs.get(cv2.CAP_PROP_FPS)

# video file to warm up
time.sleep(2.0)

# previous time for our delta time
# previous_time = time.time()

# infinite loop till video ends or 'q' is pressed
while True:
    # current frame
    ret, frame = vs.read()

    # we break the infinite loop if we have reached the end of the video
    if not ret:
        break

    countours = preprocess(frame, greenLower, isBallInUpperRegion, backgroundSubtractor)
    # check there is at least one countour
    if len(countours) > 0:
        (x, y), greenLower, isBallInUpperRegion = find_ball(frame, countours, pts)


    # orig_frame = np.copy(frame)  DEVEL

    # update timestep


    # debug on screen message
    #cv2.putText(frame, "Erode value: " + str(erodeIterations) + " Lower bound: " + str(greenLower),  (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)

    timer.update()

    # if there are enough points
    if(len(pts) > BUFFER_POINTS):
        prev_est_vel, greenLower, direction, insideCourt = draw_trajectory(frame, pts, (x, y), args, timer, camera_fps, prev_est_vel, greenLower, direction, insideCourt)

    #cv2.imshow("orig", orig_frame)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
    if key == ord('p'):
        # wait until any key is pressed
        cv2.waitKey(-1)
    #if key == ord('s'):
        # save image
        #cv2.imwrite("my_img/mid_img_2.jpg", orig_frame)

# release the camera
vs.release()

# close all windows
cv2.destroyAllWindows()
