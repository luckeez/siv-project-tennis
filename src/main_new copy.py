import cv2
import time
import numpy as np
import math

from timer import Timer
from tennis_field import TennisField
from ball import Ball
from counter import Counter

from settings import *
from utils import *

from ultralytics import YOLO

# USAGE
# Arguments:
# -v, --video: need to provide the video path. 
# -b, --buffer: max buffer size for trajectory draw.
# -y, --yolo: y/n to visualize or not yolo detection
# eg. ~/main.py --video ~/Images/tennis_match_2.mp4

args = parse_args()
args["video"] = "tennis_match_2.mp4"

# init parameters
tennis_field = TennisField(args)
timer = Timer()
ball = Ball()
counter = Counter()

model = YOLO("yolov8n.pt")

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
yolo_frame = frame1[500:950, 200:1700] # Djokovic
results = model(yolo_frame)

# infinite loop till video ends or 'q' is pressed
while True:
    counter.count_yolo+=1
    ret, frame = vs.read()

    if not ret:
        break

    if counter.count_yolo %5 == 0 and counter.count_text == 0:
        yolo_detection(frame, model, tennis_field)

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
        #cv2.putText(frame, shot, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        counter.count_text = MAX_TEXT_FRAMES
        if ball.djoko and counter.count_djoko == 0:
            counter.count_djoko = MAX_DJOKO_FRAMES 
            tennis_field.current_shot = tennis_field.shot          
    
    if counter.count_text > 0:
        cv2.putText(frame[200:900, 250:1700], "Bounce!", (int(ball.x) + 20, int(ball.y) - 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        counter.count_text -= 1        
        
    if counter.count_djoko > 0: 
        compute_shot_type(frame, counter, tennis_field, ball)

    field_2d_img = np.copy(court_img)
    cv2.circle(field_2d_img, (ball.x_2d, ball.y_2d), 3, (255, 0, 255), 2)

    # Visualize video frame
    cv2.imshow("Frame", frame)
    # Visualize 2d projection of the ball
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

# YOLO performance evaluation
# print(f"racket: {counter_racket}")
# print(f"no: {counter_no}")

# release the camera
vs.release()

# close all windows
cv2.destroyAllWindows()
