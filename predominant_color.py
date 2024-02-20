import cv2
import numpy as np

image = cv2.imread('blue_field.jpg')
ROI = {
    "minX": 400,
    "maxX": 700,
    "minY": 450,
    "maxY": 1400
}

image = image[ROI["minX"]:ROI["maxX"], ROI["minY"]:ROI["maxY"]]

def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

max_color = unique_count_app(image)
bgr_color = np.array(max_color, dtype=np.uint8)
# Convert BGR to HSV
hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]

print(hsv_color)

square_image = np.full((300, 300, 3), max_color, dtype=np.uint8)

cv2.imshow("img", image)
cv2.imshow("colo", square_image)
cv2.waitKey(0)
cv2.destroyAllWindows()