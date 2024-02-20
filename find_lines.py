import sys
import math
import cv2 as cv
import numpy as np
def main():
    
    default_file = 'my_img/blue_field.jpg'
    # Loads an image
    src = cv.imread(default_file)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    
    dst = cv.Canny(src, 100, 200, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
      
    linesP = cv.HoughLinesP(dst, 2, np.pi/180, 80, None, 600, 20)

    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            if abs(l[1] - l[3]) < 100:   # Remove horizontal lines
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,255,0), 3, cv.LINE_AA)
            else:
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    
    cv.imshow("Source", dst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
    cv.waitKey()
    return 0
    
if __name__ == "__main__":
    main()