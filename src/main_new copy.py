import cv2
import time
import numpy as np
import math

from timer import Timer
from tennis_field import TennisField
from ball import Ball

from settings import *
from utils import *

from ultralytics import YOLO

args = parse_args()
args["video"] = "tennis_match_2.mp4"

# init parameters
tennis_field = TennisField(args)
timer = Timer()
ball = Ball()

model = YOLO("yolov8n.pt")

x_pers = 0
x_racket = 0
shot = ""
shot_dir = ""

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

count_yolo = 0
count_text = 0
counter_racket = 0
counter_no = 0
count_djoko = 0
current_shot = ""

# infinite loop till video ends or 'q' is pressed
while True:
    count_yolo+=1
    ret, frame = vs.read()

    if not ret:
        break

    if count_yolo %5 == 0 and count_text == 0:
        x_racket = 0
        yolo_frame = frame[500:950, 200:1700] # Djokovic
        #cv2.imshow("yolo", yolo_frame)

        results = model(yolo_frame)

        cls = results[0].boxes.cls   # single predict

        classes = set()
        for c in cls:
            classes.add(model.names[int(c)])

        if "tennis racket" in classes:
            counter_racket += 1
        else:
            counter_no += 1

        for r in results[0]:
            if r.boxes.cls == 0:
                x_pers = int(r.boxes.xywh.T[0])
            elif r.boxes.cls == 38:
                x_racket = int(r.boxes.xywh.T[0])

        if x_racket != 0:
            if x_racket < x_pers:
                shot = "Backhand"
            else:
                shot = "Forehand"
        
        annotated_frame = results[0].plot()  # single predictz
        #cv2.imshow("Detection", annotated_frame)

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
        count_text = MAX_TEXT_FRAMES
        if ball.djoko and count_djoko == 0:
            count_djoko = MAX_DJOKO_FRAMES 
            current_shot = shot          
    
    if count_text > 0:
        cv2.putText(frame[200:900, 250:1700], "Bounce!", (int(ball.x) + 20, int(ball.y) - 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        count_text -= 1
            
        
    if count_djoko > 0:
        count_djoko -= 1
        if count_djoko == MAX_DJOKO_FRAMES-2:
            if ball.y < 250 and ball.y > 200:  
                current_shot = "Serve"
                shot_dir = ""
            elif ball.y < 200:
                current_shot = ""
                shot_dir = ""
            elif shot=="Forehand":
                if ball.x < X_THRESHOLD_LEFT:
                    if ball.dx < 0:
                        shot_dir = "Inside In"
                    elif ball.dx > SHOT_THRESHOLD_DX:
                        shot_dir = "Inside Out"
                    else:
                        shot_dir = "Center"
                elif ball.x > X_THRESHOLD_RIGHT:
                    if ball.dx < -SHOT_THRESHOLD_DX:
                        shot_dir = "Cross-court"
                    elif ball.dx > -SHOT_THRESHOLD_DX/2:
                        shot_dir = "Down the line"
                    else:
                        shot_dir = "Center"
                else:
                    shot_dir = "Center"
            elif shot == "Backhand":
                if ball.x < X_THRESHOLD_LEFT:
                    if ball.dx < SHOT_THRESHOLD_DX/2:
                        shot_dir = "Down the line"
                    elif ball.dx > SHOT_THRESHOLD_DX:
                        shot_dir = "Cross-court"
                    else:
                        shot_dir = "Center"
                elif ball.x > X_THRESHOLD_RIGHT:
                    if ball.dx < -SHOT_THRESHOLD_DX:
                        shot_dir = "Inside out"
                    elif ball.dx > 0:
                        shot_dir = "Inside In"
                    else:
                        shot_dir = "Center"
                else:
                    shot_dir = "Center"
            else:
                shot == ""

        
        var = str(f"y = {ball.y} - dx = {ball.dx}")
        
        cv2.putText(frame, current_shot, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, shot_dir, (50, 150), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, var, (50, 200), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    #    count_djoko-=1
    


    field_2d_img = np.copy(court_img)
    cv2.circle(field_2d_img, (ball.x_2d, ball.y_2d), 3, (255, 0, 255), 2)

    cv2.imshow("Frame", frame)
    #cv2.imshow("Field_2d", field_2d_img)
    
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

# print(f"racket: {counter_racket}")
# print(f"no: {counter_no}")

# release the camera
vs.release()

# close all windows
cv2.destroyAllWindows()
