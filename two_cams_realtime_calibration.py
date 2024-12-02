# import cv2
# import numpy as np
# import os
# from pathlib import Path

# # Define directories and create them if they don't exist
# Path("imagesTwoCams").mkdir(parents=True, exist_ok=True)

# # Define IP camera streams
# ip_camL = "http://192.168.137.27:8080/video"
# ip_camR = "http://192.168.137.21:8080/video"

# # Initialize cameras
# CamL = cv2.VideoCapture(ip_camL)
# # CamR = cv2.VideoCapture(ip_camR)
# CamR = cv2.VideoCapture(0)

# # Define the checkerboard dimensions
# CHECKERBOARD = (7, 10) # rows first, columns second

# '''
# Define the termination criteria of iteration in cv2.cornerSubPix

# cv2.TERM_CRITERIA_EPS : termination when algorithm has reached accuracy (epsilon) // EPS == epsilon
# cv2.TERM_CRITERIA_MAX_ITER : termination after max iterations' number
# second_arg: number max of iterations
# third_arg: the accuracy threshold, if the process achieves and improvement smaller than 0.001, algorithm terminate
# '''
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# # Prepare object points for the checkerboard


# square_size = 2.54  # Size of a square in cm (e.g., 2.54 cm = 1 inch)


# '''
# (1, CHECKERBOARD[0] * CHECKERBOARD[1], 3)
# 1 : single set of points for the checkerboard /// just one matrix
# CHECKERBOARD[0] * CHECKERBOARD[1]: total number of corners on checkerboard /// 70 lines
# 3: each point has 3 coordinates (X,Y,Z) //// 3 columns 

# initially objp are filled with zeros, with updating values of X and Y will change, but Z=0.
# '''
# objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

# '''
# generation of meshgrid of int representing grid coordinates of checkerboard

# output: Two 2D arrays:
# -- One for X-coordinates : [0,1,2,...,6] repeated for each row
# -- One for Y-coordinates : [0,0,0,...,0], [1,1,1,...,1]... [9,9,9,...,9] ...

# we take the transposes of the arrays to group X with Y
# we reshape the 2 arrays into 1 array of shape (7*10, 2)

# we assign this comuputed X and Y values to the first two dimensions and we leave Z=0
# '''
# objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)


# # Vectors to store object points and image points from both cameras
# objpoints = [] # 3D points in real world space
# imgpoints_left = [] # 2D points in image plane for the left camera
# imgpoints_right = [] # 2D points in image plane for the right camera

# # Variables for camera calibration results
# calibrated = False #  Indicates whether the camera calibration is completed.
# map1, map2 = None, None # Variables for remapping coordinates after calibration.
# mtx, dist = None, None # Camera matrix and distortion coefficients.

# while CamL.isOpened() and CamR.isOpened(): # Checks if the cameras streams are active.
#     # Read frames from both cameras / ret: bool indicates if frame has been retrieved correctly
#     retR, frameR = CamR.read()
#     retL, frameL = CamL.read()

#     if not retR or not retL:
#         print("Failed to grab frames from cameras.")
#         break

#     # Convert frames to grayscale, to simplify processing for corner detection
#     grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
#     grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)

#     # Find the chessboard corners
#     retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)
#     retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)

#     # Display the frames
#     cv2.imshow('Right Camera', frameR)
#     cv2.imshow('Left Camera', frameL)

#     if retR and retL:
#         # Refine corner locations
#         corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
#         corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)

#         # Draw and display the corners
#         cv2.drawChessboardCorners(frameR, CHECKERBOARD, corners2R, retR)
#         cv2.drawChessboardCorners(frameL, CHECKERBOARD, corners2L, retL)
#         cv2.imshow('Chessboard Right', frameR)
#         cv2.imshow('Chessboard Left', frameL)

#         # Save images when 's' is pressed
#         if cv2.waitKey(10) == ord('s'):
#             str_id_image = str(id_image)
#             print(f"Saving images {str_id_image} for both cameras.")
#             cv2.imwrite(f'imagesTwoCams/chessboard-R{id_image}.png', frameR)
#             cv2.imwrite(f'imagesTwoCams/chessboard-L{id_image}.png', frameL)

#             # Append object points and image points for calibration
#             objpoints.append(objp)
#             imgpoints_left.append(corners2L)
#             imgpoints_right.append(corners2R)

#             id_image += 1
#         else:
#             print("Images not saved.")

#     # Exit the program when the spacebar is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release cameras and close windows
# CamR.release()
# CamL.release()
# cv2.destroyAllWindows()

# # Perform stereo camera calibration
# print("Performing stereo calibration...")
# retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
# retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)

# flags = cv2.CALIB_FIX_INTRINSIC
# ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
#     objpoints, imgpoints_left, imgpoints_right,
#     mtxL, distL, mtxR, distR, grayL.shape[::-1],
#     criteria=criteria, flags=flags
# )

# print("Stereo Calibration Results:")
# print("Rotation matrix:\n", R)
# print("Translation vector:\n", T)
# print("Essential matrix:\n", E)
# print("Fundamental matrix:\n", F)

# # Stereo rectification
# rectify_scale = 1  # 0 = cropped, 1 = full image
# R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
#     mtxL, distL, mtxR, distR, grayL.shape[::-1], R, T, alpha=rectify_scale
# )

# print("Disparity-to-depth mapping matrix (Q):\n", Q)


import cv2
import numpy as np
import os
from pathlib import Path

# Define directories and create them if they don't exist
Path("imagesTwoCams").mkdir(parents=True, exist_ok=True)

# Define IP camera streams
ip_camL = "http://192.168.137.27:8080/video"
ip_camR = "http://192.168.137.21:8080/video"

# Initialize cameras
CamL = cv2.VideoCapture(ip_camL)
# CamR = cv2.VideoCapture(ip_camR)
CamR = cv2.VideoCapture(0)

# Define the checkerboard dimensions
CHECKERBOARD = (7, 10) # rows first, columns second

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
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)


# Vectors to store object points and image points from both cameras
objpoints = [] # 3D points in real world space
imgpoints_left = [] # 2D points in image plane for the left camera
imgpoints_right = [] # 2D points in image plane for the right camera

# Variables for camera calibration results
calibrated = False #  Indicates whether the camera calibration is completed.
map1, map2 = None, None # Variables for remapping coordinates after calibration.
mtx, dist = None, None # Camera matrix and distortion coefficients.

while CamL.isOpened() and CamR.isOpened(): # Checks if the cameras streams are active.
    # Read frames from both cameras / ret: bool indicates if frame has been retrieved correctly
    retR, frameR = CamR.read()
    retL, frameL = CamL.read()

    if not retR or not retL:
        print("Failed to grab frames from cameras.")
        break

    # Convert frames to grayscale, to simplify processing for corner detection
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)


    if not calibrated:

        # Find the chessboard corners
        retR, cornersR = cv2.findChessboardCorners(
            grayR, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        retL, cornersL = cv2.findChessboardCorners(
            grayL, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if retL and retR:
            # Refine corners to subpixel accuracy
            corners2R = cv2.cornerSubPix(grayR, corners2R, (11, 11), (-1, -1), criteria)
            corners2L = cv2.cornerSubPix(grayL, corners2L, (11, 11), (-1, -1), criteria)

            # Draw detected corners
            cv2.drawChessboardCorners(frameL, CHECKERBOARD, corners2L, retL)
            cv2.drawChessboardCorners(frameR, CHECKERBOARD, corners2R, retR)


            if cv2.waitKey(10) == ord('s'):
                print("Saving image and adding points for calibration.")
                objpoints.append(objp)
                imgpoints_left.append(corners2L)
                imgpoints_right.append(corners2R)
            

            if len(objpoints) >= 10:
                print("Performing camera calibration...")
                # Computes the camera matrix and distortion coefficients.
                retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
                retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)
                
                if retL and retR:
                    print("Individual calibration successful. Performing stereo calibration...")

                    ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
                        objpoints, imgpoints_left, imgpoints_right,
                        mtxL, distL, mtxR, distR,
                        grayL.shape[::-1],
                        criteria=criteria,
                        flags=cv2.CALIB_FIX_INTRINSIC
                    )

                    if ret:
                        print("Stereo calibration successful!")
                        calibrated = True
                    else: print("Stereo calibration failed. Collect more data.")
    else:
        RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(
            mtxL, distL, mtxR, distR, grayL.shape[::-1], R, T
        )

        mapLx, mapLy = cv2.initUndistortRectifyMap(mtxL, distL, RL, PL, grayL.shape[::-1], cv2.CV_32FC1)
        mapRx, mapRy = cv2.initUndistortRectifyMap(mtxR, distR, RR, PR, grayR.shape[::-1], cv2.CV_32FC1)

        undistortedL_frame = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
        undistortedR_frame = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

        # Detect the checkerboard in the undistorted frame
        retL, cornersL = cv2.findChessboardCorners(
            cv2.cvtColor(undistortedL_frame, cv2.COLOR_BGR2GRAY), CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        retR, cornersR = cv2.findChessboardCorners(
            cv2.cvtColor(undistortedR_frame, cv2.COLOR_BGR2GRAY), CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if retL and retR:
            corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            
            cv2.drawChessboardCorners(undistortedL_frame, CHECKERBOARD, corners2L, ret)
            cv2.drawChessboardCorners(undistortedR_frame, CHECKERBOARD, corners2R, ret)

            # Estimate pose of the checkerboard (rotation rvec and translation tvec).
            _, rvecL, tvecL = cv2.solvePnP(objp, corners2L, mtxL, distL)
            _, rvecR, tvecR = cv2.solvePnP(objp, corners2R, mtxR, distR)

            # Calculate distance
            xL, yL, zL = tvecL.flatten()
            distance = np.sqrt(xL**2 + yL**2 + zL**2)
            
            xR, yR, zR = tvecR.flatten()
            distance = np.sqrt(xR**2 + yR**2 + zR**2)

            # Display the distance on the frame
            cv2.putText(
                undistortedL_frame, f"Distance: {distance:.2f} cm",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            cv2.putText(
                undistortedR_frame, f"Distance: {distance:.2f} cm",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

        cv2.imshow('Camera Left (Undistorted)', undistortedL_frame)
        cv2.imshow('Camera Right (Undistorted)', undistortedR_frame)

    # Display the frames
    cv2.imshow('Left Camera', frameL)
    cv2.imshow('Right Camera', frameR)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

CamL.release()
CamR.release()
cv2.destroyAllWindows()
