import cv2 
import numpy as np

ROI = {
    "minX": 200,
    "maxX": 900,
    "minY": 250,
    "maxY": 1700
}

img = cv2.imread("my_img/mid_img_2.jpg")
img_2d = cv2.imread('my_img/tennis_court_small.jpg') 

frame_roi = img[ROI["minX"]:ROI["maxX"], ROI["minY"]:ROI["maxY"]]

cv2.imshow("court", frame_roi)
cv2.imshow("ball", img_2d)
cv2.waitKey(0)

cv2.destroyAllWindows()