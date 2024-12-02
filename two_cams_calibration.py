import cv2
import numpy as np
import glob
from pathlib import Path


# Define directories and create them if they don't exist
Path("imagesTwoCams").mkdir(parents=True, exist_ok=True)

# Define IP camera streams
ip_camL = "http://192.168.137.27:8080/video"
ip_camR = "http://192.168.137.21:8080/video"

# Initialize cameras
CamL = cv2.VideoCapture(ip_camL)
CamR = cv2.VideoCapture(ip_camR)
# CamR = cv2.VideoCapture(0)


# ID for saved images
id_image = 0

while CamL.isOpened() and CamR.isOpened():
    # Read frames from both cameras
    retR, frameR = CamR.read()
    retL, frameL = CamL.read()

    if not retR or not retL:
        print("Failed to grab frames from cameras.")
        break

    cv2.imshow('Chessboard Right', frameR)
    cv2.imshow('Chessboard Left', frameL)

    # # Save images when 's' is pressed
    if cv2.waitKey(10) == ord('s'):
    #     str_id_image = str(id_image)
    #     print(f"Saving images {str_id_image} for both cameras.")
        cv2.imwrite(f'imagesTwoCams/chessboard-R{id_image}.png', frameR)
        cv2.imwrite(f'imagesTwoCams/chessboard-L{id_image}.png', frameL)
    #     # Append object points and image points for calibration
        id_image += 1

    # else:
    #     print("Images not saved.")

    # Exit the program when the spacebar is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release cameras and close windows
CamR.release()
CamL.release()
cv2.destroyAllWindows()



# # Define the checkerboard dimensions
# CHECKERBOARD = (7, 10)
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# # Vectors to store object points and image points from both cameras
# objpoints = []
# imgpoints_left = []
# imgpoints_right = []




# # Prepare object points for the checkerboard
# square_size = 2.54  # Size of a square in cm (e.g., 2.54 cm = 1 inch)
# objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size




# # Extract images from the directory
# images = glob.glob('./imagesTwoCams/*.png')
# for fname in images:
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Find the chessboard corners
#     ret, corners = cv2.findChessboardCorners(
#         gray, CHECKERBOARD, 
#         cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
#     )
    
#     if ret:
#         objpoints.append(objp)
#         # Refine pixel coordinates for given 2D points
#         corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#         imgpoints.append(corners2)
        
#         # Draw and display the corners
#         img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
#         cv2.imshow('Calibration', img)
#         cv2.waitKey(0)

# cv2.destroyAllWindows()

# # Perform camera calibration
# h, w = gray.shape[:2]
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# # Print camera calibration results
# print("Camera matrix:")
# print(mtx)
# print("Distortion coefficients:")
# print(dist)

# # Calculate and display distances for each image
# for i, tvec in enumerate(tvecs):
#     x, y, z = tvec.flatten()
#     distance = np.sqrt(x**2 + y**2 + z**2)  # Euclidean distance
#     print(f"\nCheckerboard position for image {i + 1}:")
#     print(f"X: {x:.2f} cm, Y: {y:.2f} cm, Z: {z:.2f} cm")
#     print(f"Euclidean Distance: {distance:.2f} cm")





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
