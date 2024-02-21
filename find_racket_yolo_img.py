from ultralytics import YOLO
from PIL import Image
import cv2
import torchvision
import time
import numpy as np

model = YOLO("yolov8n.pt")

frame = cv2.imread('my_img/mid_img_2.jpg') 
# img = torchvision.io.read_image("./sample.jpg")

roi_frame = frame[500:950, 200:1700]

results = model(roi_frame)

x1 = int(results[0][0].boxes.xyxy.T[0])
y1 = int(results[0][0].boxes.xyxy.T[1])
x2 = int(results[0][0].boxes.xyxy.T[2])
y2 = int(results[0][0].boxes.xyxy.T[3])

person_frame = roi_frame[y1-30:y2+30, x1-100:x2+100]

results_racket = model(person_frame)

cls = results_racket[0].boxes.cls
classes = set()
for c in cls:
    classes.add(model.names[int(c)])

annotated_frame = results_racket[0].plot()
cv2.imshow("Detection", annotated_frame)

# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#     im.show()  # show image


cv2.imshow("Frame", frame)
cv2.waitKey(0)

cv2.destroyAllWindows()
