import cv2
import numpy as np
import os
from pathlib import Path

# Define directories and create them if they don't exist
Path("imagesTwoCams").mkdir(parents=True, exist_ok=True)

# Define IP camera streams
ip_camL = "http://192.168.1.41:8080/video"
ip_camR = "http://192.168.1.36:8080/video"

# Initialize cameras
CamL = cv2.VideoCapture(ip_camL)
CamR = cv2.VideoCapture(ip_camR)

# Define the checkerboard dimensions
CHECKERBOARD = (7, 10)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for the checkerboard
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Vectors to store object points and image points from both cameras
objpoints = []
imgpoints_left = []
imgpoints_right = []

# ID for saved images
id_image = 0

while True:
    # Read frames from both cameras
    retR, frameR = CamR.read()
    retL, frameL = CamL.read()

    if not retR or not retL:
        print("Failed to grab frames from cameras.")
        break

    # Convert frames to grayscale
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)
    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)

    # Display the frames
    cv2.imshow('Right Camera', frameR)
    cv2.imshow('Left Camera', frameL)

    if retR and retL:
        # Refine corner locations
        corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        cv2.drawChessboardCorners(frameR, CHECKERBOARD, corners2R, retR)
        cv2.drawChessboardCorners(frameL, CHECKERBOARD, corners2L, retL)
        cv2.imshow('Chessboard Right', frameR)
        cv2.imshow('Chessboard Left', frameL)

        # Save images when 's' is pressed
        if cv2.waitKey(0) & 0xFF == ord('s'):
            str_id_image = str(id_image)
            print(f"Saving images {str_id_image} for both cameras.")
            cv2.imwrite(f'imagesTwoCams/chessboard-R{id_image}.png', frameR)
            cv2.imwrite(f'imagesTwoCams/chessboard-L{id_image}.png', frameL)

            # Append object points and image points for calibration
            objpoints.append(objp)
            imgpoints_left.append(corners2L)
            imgpoints_right.append(corners2R)

            id_image += 1
        else:
            print("Images not saved.")

    # Exit the program when the spacebar is pressed
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# Release cameras and close windows
CamR.release()
CamL.release()
cv2.destroyAllWindows()

# Perform stereo camera calibration
print("Performing stereo calibration...")
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)

flags = cv2.CALIB_FIX_INTRINSIC
ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtxL, distL, mtxR, distR, grayL.shape[::-1],
    criteria=criteria, flags=flags
)

print("Stereo Calibration Results:")
print("Rotation matrix:\n", R)
print("Translation vector:\n", T)
print("Essential matrix:\n", E)
print("Fundamental matrix:\n", F)

# Stereo rectification
rectify_scale = 1  # 0 = cropped, 1 = full image
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, grayL.shape[::-1], R, T, alpha=rectify_scale
)

print("Disparity-to-depth mapping matrix (Q):\n", Q)
