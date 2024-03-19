import cv2
import time
import numpy as np
import math

from timer import Timer
from tennis_field import TennisField
from ball import Ball

from settings import *
from utils import *

args = parse_args()
args["video"] = "tennis_match_2.mp4"

# init parameters
tennis_field = TennisField(args)
timer = Timer()
ball = Ball()

court_img = cv2.imread('my_img/tennis_court_small.jpg') 

# substracts the background from the frame
backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=75, detectShadows=False)

# video reference from the arguments
vs = cv2.VideoCapture(tennis_field.args["video"])

# get video fps
camera_fps = vs.get(cv2.CAP_PROP_FPS)

# video file to warm up
time.sleep(2.0)

ret, frame1 = vs.read()

court_mask_points = find_court_mask_points(frame1)

# infinite loop till video ends or 'q' is pressed
while True:
    ret, frame = vs.read()

    if not ret:
        break

    countours = preprocess(frame, tennis_field, backgroundSubtractor)
    # check there is at least one countour
    if len(countours) > 0:
        find_ball(frame, countours, tennis_field, ball)

    timer.update()

    draw_field_lines(frame, court_mask_points)

    # if there are enough points
    if(len(tennis_field.pts) > BUFFER_POINTS):
        draw_trajectory(frame, tennis_field, ball, timer, court_mask_points)
    
    if ball.bounce:
        cv2.circle(court_img, (ball.x_2d, ball.y_2d), 3, (0, 255, 0), 2)

    field_2d_img = np.copy(court_img)
    cv2.circle(field_2d_img, (ball.x_2d, ball.y_2d), 3, (255, 0, 255), 2)

    

    cv2.imshow("Frame", frame)
    cv2.imshow("Field_2d", field_2d_img)

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
