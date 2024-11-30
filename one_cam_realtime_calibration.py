import cv2
import numpy as np
import os
from pathlib import Path

# Define directories and create them if they don't exist
Path("imagesOneCam").mkdir(parents=True, exist_ok=True)

# Define the IP camera stream
ip_cam = "http://192.168.51.81:8080/video"  # Replace with your camera IP
Cam = cv2.VideoCapture(ip_cam)

# Define the checkerboard dimensions
CHECKERBOARD = (7, 10)  # Columns first, rows second
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for the checkerboard
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Vectors to store object points and image points from the camera
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# ID for saved images
id_image = 0

while True:
    # Read frame from the camera
    ret, frame = Cam.read()

    if not ret:
        print("Failed to grab frame from the camera.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # Display the frame
    cv2.imshow('Camera', frame)

    if ret:
        # Refine corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
        cv2.imshow('Chessboard', frame)

        # Save images when 's' is pressed
        if cv2.waitKey(0) & 0xFF == ord('s'):
            str_id_image = str(id_image)
            print(f"Saving image {str_id_image} from the camera.")
            cv2.imwrite(f'imagesOneCam/chessboard-{id_image}.png', frame)

            # Append object points and image points for calibration
            objpoints.append(objp)
            imgpoints.append(corners2)

            id_image += 1
        else:
            print("Image not saved.")

    # Exit the program when the spacebar is pressed
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# Release the camera and close windows
Cam.release()
cv2.destroyAllWindows()

# Perform single camera calibration
print("Performing single camera calibration...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera Calibration Results:")
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)
print("Rotation vectors:\n", rvecs)
print("Translation vectors:\n", tvecs)
