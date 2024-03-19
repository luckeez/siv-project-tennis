import sys
import math
import cv2 as cv
import numpy as np
def main():


    ROI = {
        "minX": 200,
        "maxX": 900,
        "minY": 250,
        "maxY": 1700
    }
    
    default_file = 'my_img/mid_img_2.jpg'
    #default_file = 'my_img/blue_field.jpg'
    # Loads an image
    image = cv.imread(default_file)
    # Check if image is loaded fine

    src = image[ROI["minX"]:ROI["maxX"], ROI["minY"]:ROI["maxY"]]
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    src = cv.GaussianBlur(src, (5, 5), 0)
    
    dst = cv.Canny(src, 100, 200, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
      
    linesP = cv.HoughLinesP(dst, 1, np.pi/180, 80, None, 400, 20)

    field_lines = []
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,255,0), 3, cv.LINE_AA)
            if abs(l[1] - l[3]) < 100:   # Remove horizontal lines
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, cv.LINE_AA)
            else:
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
                field_lines.append(l)
                print(field_lines)
    
    linesP



    xl = int(field_lines[0][2] - ((field_lines[0][2] - field_lines[0][0]) * (field_lines[1][3] - field_lines[0][3])) / (field_lines[0][1] - field_lines[0][3]))
    cv.line(cdstP, (field_lines[0][2], field_lines[0][3]), (xl, field_lines[1][3]), (0,255,0), 3, cv.LINE_AA)

    xr = int(field_lines[1][2] - ((field_lines[1][2] - field_lines[1][0]) * (field_lines[1][3] - field_lines[0][3])) / (field_lines[1][3] - field_lines[1][1]))
    cv.line(cdstP, (field_lines[1][2], field_lines[1][3]), (xr, field_lines[0][3]), (0,255,0), 3, cv.LINE_AA)

    p1 = (field_lines[0][2], field_lines[0][3])
    p2 = (xl, field_lines[1][3])
    p3 = (xr, field_lines[0][3])
    p4 = (field_lines[1][2], field_lines[1][3])
    
    court_points = np.array([p1, p2, p3, p4])

    print(court_points)

    COURT_MASK_POINTS = np.array([(267, 141), (25, 681), (1422, 686), (1161, 144)])

    # cv.line(cdstP, (633, 315), (460, 844), (0,255,0), 3, cv.LINE_AA)
    # cv.line(cdstP, (1475, 844), (1282, 315), (0,255,0), 3, cv.LINE_AA)

    # cv.line(cdstP, (267, 141), (25, 681), (0,255,0), 3, cv.LINE_AA)
    # cv.line(cdstP, (1475, 844), (1161, 144), (0,255,0), 3, cv.LINE_AA)
    
    # pts = np.array([ [field_lines[0][0], field_lines[0][1]], 
    #                 [field_lines[0][2], field_lines[0][3]],
    #                 [field_lines[1][0], field_lines[1][1]],
    #                 [field_lines[1][2], field_lines[0][1]] ],
    #                 np.int32)
    

    # mask = np.zeros(src.shape[:2], dtype="uint8")
    # cv.fillConvexPoly(mask, pts, color=(255,255,255))
    # masked = cv.bitwise_and(src, src, mask=mask)

    # cv.imshow("Fil", masked)
    
    cv.imshow("Source", dst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
    cv.waitKey()
    return 0
    
if __name__ == "__main__":
    main()