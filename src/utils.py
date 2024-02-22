import cv2
import numpy as np
import math
import time

from settings import *


def preprocess(frame, greenLower, isBallInUpperRegion, backgroundSubtractor):

    roi_frame = frame[ROI["minX"]:ROI["maxX"], ROI["minY"]:ROI["maxY"]]
    cv2.imshow("ROI", roi_frame)

    # we blur our frame for possible noise
    blurred = cv2.GaussianBlur(frame, (ODD_BLUR, ODD_BLUR), 0)
    #cv2.imshow("Blurred image", blurred)

    # set our frame to hsv color
    hsv_blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv blurred", hsv_blurred)

    # To filter the players we take the range from our thresholds we defined
    mask = cv2.inRange(hsv_blurred, greenLower, GREEN_UPPER)
    #cv2.imshow("filtered mask", mask)

    # Because the top region the ball is smaller from the camera perspective we erode less than in the lower region
    erodeIterations = 2 if not isBallInUpperRegion else 1

    mask = cv2.erode(mask, None, iterations=erodeIterations)
    #cv2.imshow("eroded mask", mask)

    mask = cv2.dilate(mask, None, iterations=2)
    #print(str(greenLower) + " " + str(GREEP_UPPER))
    #cv2.imshow("dilated mask", mask)

    # We apply the substractor to our blurred frame
    bkgnMask = backgroundSubtractor.apply(blurred)
    #cv2.imshow("the substractor", bkgnMask)

    # We define ROI (region of interest)
    mask = mask[ROI["minX"]:ROI["maxX"], ROI["minY"]:ROI["maxY"]]
    bkgnMask = bkgnMask[ROI["minX"]:ROI["maxX"], ROI["minY"]:ROI["maxY"]]

    # since we just want the region with white pixels (no gray) we set our threshold to 254, 255
    _, thresholdMask = cv2.threshold(bkgnMask, 254, 255, cv2.THRESH_BINARY)
    #cv2.imshow("threshold", thresholdMask)

    thresholdMask = cv2.erode(thresholdMask, None, iterations=1)
    #cv2.imshow("threshold eroded", thresholdMask)

    thresholdMask = cv2.dilate(thresholdMask, None, iterations=3)
    #cv2.imshow("threshold dilated", thresholdMask)

    # we find out countours from our threshold mask
    #countours_thresh, _ = cv2.findContours(
    #    thresholdMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # we apply a 'and' operator to our mask (which filters the playesr) and our thresholdMask which deleted the background and keeps the moving objects
    result = mask & thresholdMask
    #cv2.imshow("Result mask", result)

    # from our result we look up for the countours
    countours, _ = cv2.findContours(
        result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return countours





def find_ball(frame, countours, pts):

    # values to define the three areas of our video.
    frame_height = frame[ROI["minX"]:ROI["maxX"], ROI["minY"]:ROI["maxY"]].shape[0]/3
    frame_height_middle_region = frame_height*2

    # we get the ball coordinates and radius
    c = max(countours, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)

    # we define the green lower threshold depending on where the ball was last seen
    if y < frame_height-OFFSET and y < frame_height_middle_region:
        greenLower = TOP_GREEN_LOWER
    elif y > frame_height-OFFSET and y < frame_height_middle_region:
        greenLower = MIDDLE_GREEN_LOWER
    else:
        greenLower = BOTTOM_GREEN_LOWER

    # Check if the ball is on the upper or lower region
    isBallInUpperRegion = True if y < frame_height_middle_region else False

    # If its a valid point we save the value and print some visual information on screen
    if M["m00"] != 0:
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 1:
            frame_roi = frame[ROI["minX"]:ROI["maxX"], ROI["minY"]:ROI["maxY"]]
            cv2.circle(frame_roi, (int(x), int(y)),
                        int(radius), (0, 255, 255), 2)
            cv2.putText(frame_roi, "Radius: " + str(int(radius)) + " X,Y: " + str(int(x)) + " " + str(
                int(y)), (int(x) + 20, int(y) + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            pts.appendleft(center)

    return (x, y), greenLower, isBallInUpperRegion



def draw_trajectory(frame, pts, pos, args, timer, camera_fps, prev_est_vel, greenLower, direction, insideCourt):
    est_vel = [0,0]
    (x, y) = pos

    for i in np.arange(1, len(pts)):
        
        # draw line for vizual ball tracking on video
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame[ROI["minX"]:ROI["maxX"], ROI["minY"]:ROI["maxY"]]
                    , pts[i - 1], pts[i], (0, 0, 255), thickness)

        # find the current direction of the ball
        if i == 1:
            # compute the difference between the x and y
            #dX = pts[i-BUFFER_POINTS][0] - pts[i][0]
            #dY = pts[i-BUFFER_POINTS][1] - pts[i][1]

            dX = pts[0][0] - pts[BUFFER_POINTS][0]
            dY = pts[0][1] - pts[BUFFER_POINTS][1]


            # clean direction messages
            (dirX, dirY) = ("", "")
            # check if there is movement in x-direction
            if np.abs(dX) > 3:
                dirX = "East" if np.sign(dX) == 1 else "West"
            # check if there is movement in y-direction
            if np.abs(dY) > 3:
                dirY = "South" if np.sign(dY) == 1 else "North"
                greenLower = BOTTOM_GREEN_LOWER if np.sign(
                    dY) == 1 else BOTTOM_GREEN_LOWER

            # handle when both directions are non-empty
            if dirX != "" and dirY != "":
                direction = "{}-{}".format(dirY, dirX)
            # otherwise, only one direction is non-empty
            else:
                direction = dirX if dirX != "" else dirY

            # we estimate the velocity of the ball to see if it bounced
            if timer.camera_timer > (1.0 / camera_fps):    # INUTILE
                # estimate velocity
                est_vel[0] = dX / timer.dt
                est_vel[1] = dY / timer.dt

                # check if the sign of the velocity has changed
                if np.sign(est_vel[0]) != np.sign(prev_est_vel[0]) or np.sign(est_vel[1]) != np.sign(prev_est_vel[1]):
                    dvx = abs(est_vel[0] - prev_est_vel[0])
                    dvy = abs(est_vel[1] - prev_est_vel[1])
                    change_vel = math.sqrt(dvx*dvx + dvy*dvy)
                    if change_vel > BOUNCE_TRESH:
                        ballInsideOutsideTest = cv2.pointPolygonTest(
                            COURT_MASK_POINTS, (x, y), False)
                        # -1 is outside, 1 is inside and 0 is on the contour
                        # outside
                        insideCourt = ""
                        if ballInsideOutsideTest == -1:
                            insideCourt += "Out"
                        else:
                            insideCourt += "Inside"

                        cv2.putText(frame[200:900,
                                            250:1700], "Bounce!", (int(x) + 20, int(y) - 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                                    (0, 255, 0), 2)

                # update previous state trackers
                prev_est_vel = est_vel[:]

                # reset camera timer
                timer.camera_timer = 0
            else:
                print("ciao")

    cv2.putText(frame, direction + " dx: {}, dy: {}:".format(dX, dY), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 255), 3)
    cv2.putText(frame, insideCourt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 0), 3)
    

    return prev_est_vel, greenLower, direction, insideCourt



def update_timestep(previous_time, camera_timer):
    now_time = time.time()
    # we have the delta time difference between frames
    dt = now_time - previous_time  # temporal distance between two succeding frames (more or less constant)
    dt *= TIME_SCALE
    previous_time = now_time
    camera_timer += dt 

    return previous_time, camera_timer