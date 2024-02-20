import cv2
import numpy as np
from operator import itemgetter


SOGLIA = 10

ROI = {
    "minX": 200,
    "maxX": 900,
    "minY": 250,
    "maxY": 1700
}
COURT_MASK_POINTS = np.array([(267, 141), (25, 681), (1422, 686), (1161, 144)])

def find_field_color(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    max_color = colors[count.argmax()]
    bgr_color = np.array(max_color, dtype=np.uint8)
    hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
    return hsv_color



# Leggi l'immagine
image = cv2.imread('my_img/original_img.jpg')
field_color = find_field_color(image[400:700, 450:1400])
#image = image[ROI["minX"]:ROI["maxX"], ROI["minY"]:ROI["maxY"]]

# Converte l'immagine in spazio colore HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definisci i limiti del colore del campo da tennis nel formato HSV
lower_bound = np.array([field_color[0]-8, field_color[1]-30, 0])  # Valori di H, S, V minimi
upper_bound = np.array([field_color[0]+8, field_color[1]+30, 255])  # Valori di H, S, V massimi

# Crea una maschera per isolare il colore del campo da tennis
mask = cv2.inRange(hsv, lower_bound, upper_bound)
mask = cv2.erode(mask, None, iterations=5)

index = 40
mask = cv2.dilate(mask, None, iterations=index)
mask = cv2.erode(mask, None, iterations=index-5)

# Applica la maschera all'immagine originale
result = cv2.bitwise_and(image, image, mask=mask)

# Converte l'immagine risultante in scala di grigi
gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
#cv2.imshow("res", gray_result)
#cv2.waitKey(0)

# Applica il rilevamento dei contorni utilizzando Canny
edges = cv2.Canny(gray_result, 50, 150)

# Trova i contorni nell'immagine
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Disegna i contorni sull'immagine originale
cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=2)

c = max(contours, key=cv2.contourArea)

extreme_points = {}
# Extreme points: double field, I have to find the single ones
extreme_points["maxX"] = (tuple(c[c[:, :, 0].argmax()][0])) # maxX
extreme_points["minX"] = (tuple(c[c[:, :, 0].argmin()][0])) # minX
extreme_points["minY"] = (tuple(c[c[:, :, 1].argmin()][0])) # minY
extreme_points["maxY"] = (tuple(c[c[:, :, 1].argmax()][0])) # maxY

field = {}
top_points = []

for p in c:
    p = p[0]
    # bottom-left
    if abs(p[0]-extreme_points["minX"][0]) <= SOGLIA and abs(p[1]-extreme_points["maxY"][1]) <=SOGLIA and not "bottom-left" in field:
        field["bottom-left"] = p
    # bottom-right
    if abs(p[0]-extreme_points["maxX"][0]) <= SOGLIA and abs(p[1]-extreme_points["maxY"][1]) <=SOGLIA and not "bottom-right" in field:
        field["bottom-right"] = p

    if abs(p[1] - extreme_points["minY"][1]) <= 3:
        top_points.append(p)
    '''
    # top-right
    if abs(p[0][0]-extreme_points[2][0]) <= SOGLIA and abs(p[0][1]-extreme_points[2][1]) <=SOGLIA and not "top-right" in field:
        field["top-right"] = p[0]
    # top-left
    if abs(p[0][0]-extreme_points[3][0]) <= SOGLIA and abs(p[0][1]-extreme_points[2][1]) <=SOGLIA and not "top-left" in field:
        field["top-left"] = p[0]
    '''

field["top-right"] = max(top_points, key=itemgetter(0))
field["top-left"] = min(top_points, key=itemgetter(0))

for p in extreme_points.values():
    cv2.circle(image, p, 5, (0, 0, 255), -1)

for p in field.values():
    cv2.circle(image, p, 5, (255, 0, 0), -1)

# Visualizza l'immagine con i contorni del campo da tennis
cv2.imshow('Tennis Field Edges', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
