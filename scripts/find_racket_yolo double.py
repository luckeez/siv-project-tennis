from ultralytics import YOLO
import cv2
import time
import numpy as np

model = YOLO("yolov8n.pt")

# performance evaluation
counter_racket = 0
counter_no = 0

# img = cv2.imread('my_img/mid_img.jpg') 
# img = torchvision.io.read_image("./sample.jpg")

vs = cv2.VideoCapture("tennis_match_2.mp4")

# video file to warm up
time.sleep(2.0)


while True:
    # current frame
    ret, frame = vs.read()

    # we break the infinite loop if we have reached the end of the video
    if not ret:
        break

    roi_frame = frame[500:950, 200:1700] # Djokovic
    #roi_frame = frame[150:400, 450:1450] # Rune

    results = model(roi_frame)

# **** double predict ****

    x1 = int(results[0][0].boxes.xyxy.T[0])
    y1 = int(results[0][0].boxes.xyxy.T[1])
    x2 = int(results[0][0].boxes.xyxy.T[2])
    y2 = int(results[0][0].boxes.xyxy.T[3])

    person_frame = roi_frame[0:y2+30, x1-100:x2+100]

    results_racket = model(person_frame)

# ***************************

    cls = results_racket[0].boxes.cls  # double predict
    #cls = results[0].boxes.cls   # single predict

    classes = set()
    for c in cls:
        classes.add(model.names[int(c)])

    if "tennis racket" in classes:
        counter_racket += 1
    else:
        counter_no += 1

    #annotated_frame = results[0].plot()  # single predict
    annotated_frame = results_racket[0].plot()   # double predict
    cv2.imshow("Detection", annotated_frame)

    # for r in results:
    #     im_array = r.plot()  # plot a BGR numpy array of predictions
    #     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    #     im.show()  # show image


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
    if key == ord('p'):
        # wait until any key is pressed
        cv2.waitKey(-1)

# performance evaluation
print(f"Rackets: {counter_racket}")
print(f"No racket: {counter_no}")


# release the camera
vs.release()

# close all windows
cv2.destroyAllWindows()
