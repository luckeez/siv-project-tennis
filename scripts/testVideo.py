import cv2
import time
import numpy as np
import math
from matplotlib import pyplot as plt

video_path = "tennis_match_2.mp4"
blur_factor = 5
#max_frame_array = [200, 900, 250, 1700]

max_frame = {
    "minX": 200,
    "maxX": 900,
    "minY": 250,
    "maxY": 1700
}

greenLower = (30, 55, 150)
# top
topGreenLower = (20, 0, 150)
# middle
middleGreenLower = (45, 35, 150)
# bottom
bottomGreenLower = (35, 55, 150)
# initial upper
greenUpper = (65, 255, 255)

backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=99, detectShadows=False)

vs = cv2.VideoCapture(video_path)

time.sleep(2)

if not vs.isOpened():
    print("Errore nell'apertura del file")
else:
    while True:
        ret, frame = vs.read()  # frame is my image
        if not ret:
            break

        blurred = cv2.GaussianBlur(frame, (blur_factor, blur_factor), 0)
        hsv_blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_blurred, greenLower, greenUpper)
        #erodeIterations = 2 if not isBallInUpperRegion else 1

        mask = cv2.erode(mask, None, iterations=1)
        #cv2.imshow("eroded mask", mask)
        #cv2.waitKey(0)

        mask = cv2.dilate(mask, None, iterations=2)
        bkgnMask = backgroundSubtractor.apply(blurred)
        bkgnMask = bkgnMask[max_frame["minX"]:max_frame["maxX"], max_frame["minY"]:max_frame["maxY"]]
        _, thresholdMask = cv2.threshold(bkgnMask, 254, 255, cv2.THRESH_BINARY)
        thresholdMask = cv2.erode(thresholdMask, None, iterations=1)
        thresholdMask = cv2.dilate(thresholdMask, None, iterations=3)
        cv2.imshow("the substractor", bkgnMask)
        cv2.imshow("threshold", thresholdMask)
        key = cv2.waitKey(1) & 0xFF

            # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break
        if key == ord('p'):
        # wait until any key is pressed
            cv2.waitKey(-1)


        #plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))


        



        

