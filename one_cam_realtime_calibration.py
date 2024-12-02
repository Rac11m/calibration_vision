import cv2
import numpy as np
from pathlib import Path

# Define directory and create it if it doesn't exist
Path("imagesOneCam").mkdir(parents=True, exist_ok=True)

# Define the IP camera stream
ip_cam = "http://192.168.137.27:8080/video" 

# Define the videoCapture  
# Cam = cv2.VideoCapture(ip_cam)
Cam = cv2.VideoCapture(0)

# Define the checkerboard dimensions
CHECKERBOARD = (7, 10)  # rows first, columns second

'''
Define the termination criteria of iteration in cv2.cornerSubPix

cv2.TERM_CRITERIA_EPS : termination when algorithm has reached accuracy (epsilon) // EPS == epsilon
cv2.TERM_CRITERIA_MAX_ITER : termination after max iterations' number
second_arg: number max of iterations
third_arg: the accuracy threshold, if the process achieves and improvement smaller than 0.001, algorithm terminate
'''
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for the checkerboard


square_size = 2.54  # Size of a square in cm (e.g., 2.54 cm = 1 inch)

'''
(1, CHECKERBOARD[0] * CHECKERBOARD[1], 3)
1 : single set of points for the checkerboard /// just one matrix
CHECKERBOARD[0] * CHECKERBOARD[1]: total number of corners on checkerboard /// 70 lines
3: each point has 3 coordinates (X,Y,Z) //// 3 columns 

initially objp are filled with zeros, with updating values of X and Y will change, but Z=0.
'''
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

'''
generation of meshgrid of int representing grid coordinates of checkerboard

output: Two 2D arrays:
-- One for X-coordinates : [0,1,2,...,6] repeated for each row
-- One for Y-coordinates : [0,0,0,...,0], [1,1,1,...,1]... [9,9,9,...,9] ...

we take the transposes of the arrays to group X with Y
we reshape the 2 arrays into 1 array of shape (7*10, 2)

we assign this comuputed X and Y values to the first two dimensions and we leave Z=0
'''
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size


# Vectors to store object points and image points from the camera
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Variables for camera calibration results
calibrated = False #  Indicates whether the camera calibration is completed.
map1, map2 = None, None # Variables for remapping coordinates after calibration.
mtx, dist = None, None # Camera matrix and distortion coefficients.

while Cam.isOpened(): # Checks if the camera stream is active.
    ret, frame = Cam.read() # Reads a frame from the camera. / ret: bool indicates if frame has been retrieved correctly 

    if not ret:
        print("Failed to grab frame from the camera.")
        break

    # convert the frame to grayscale, to simplify processing for corner detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 


    if not calibrated:
        # detection of internal corners of the checkerboard
        '''
            Args:
                - grayscale frame, because findChessboardCorners works on intensity values (the primary information stored within pixels).
                - checkerboard dimensions
                - flags:
                        * CALIB_CB_ADAPTIVE_THRESH: Adaptative thresholding for better corner detection in varying lighting conditions (ensures robust detection in varying lighting.)
                        * CALIB_CB_FAST_CHECK: Performs a quick rejection test to eliminate unlikely regions, speeding up detection. (speeds up the process, especially useful for real-time applications.)
                        * CALIB_CB_NORMALIZE_IMAGE: Normalize the image's brightness and constrast to enhance corner detection (enhances contrast, improving corner detection accuracy.) 

            Returned values:
            ret : bool 
                if true -> corners found
                else -> not found
            corners: an array of detected corner points in the image
                shape (CHECKERBOARD[0] * CHECKERBOARD[1], 1, 2) // lines: 70, columns: 2 (x,y) // second_arg means 1 matrix 
        '''
        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            # Refines corner accuracy to subpixel level.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draws the detected corners on the frame.
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)

            if cv2.waitKey(10) == ord('s'):
                print("Saving image and adding points for calibration.")
                objpoints.append(objp)
                imgpoints.append(corners2)

            if len(objpoints) >= 10:
                print("Performing camera calibration...")
                # Computes the camera matrix and distortion coefficients.
                ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                if ret:
                    # Prepares maps for real-time undistortion.
                    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, gray.shape[::-1], cv2.CV_32FC1)
                    calibrated = True
                else:
                    print("Calibration failed. Collect more data.")

    else:
        # Apply undistortion / Corrects distortion using the calibration maps.
        undistorted_frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        # Detect the checkerboard in the undistorted frame
        ret, corners = cv2.findChessboardCorners(
            cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY), CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(undistorted_frame, CHECKERBOARD, corners2, ret)

            # Estimate pose of the checkerboard (rotation rvec and translation tvec).
            _, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)

            # Calculate distance
            x, y, z = tvec.flatten()
            distance = np.sqrt(x**2 + y**2 + z**2)

            # Display the distance on the frame
            cv2.putText(
                undistorted_frame, f"Distance: {distance:.2f} cm",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

        cv2.imshow('Undistorted Camera', undistorted_frame)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

Cam.release()
cv2.destroyAllWindows()
