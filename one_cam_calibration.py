import cv2
import numpy as np
import glob
from pathlib import Path

# Create directory for storing images
Path("imagesOneCam").mkdir(parents=True, exist_ok=True)

# IP camera stream
ip_cam = "http://192.168.51.81:8080/video"

# Open the camera stream
cam = cv2.VideoCapture(ip_cam)
num = 0

# Capture images from the camera
while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break
    cv2.imshow("Smartphone Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) == ord('s'):
        cv2.imwrite('imagesOneCam/img' + str(num) + '.png', frame)
        num += 1

cam.release()
cv2.destroyAllWindows()

# Calibration with checkerboard
# Define the dimensions of the checkerboard
CHECKERBOARD = (7, 10)  # Columns x Rows

# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Vectors to store 3D points and 2D points for each image
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Define the world coordinates for 3D points
square_size = 2.54  # Size of a square in cm (e.g., 2.54 cm = 1 inch)
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

# Extract images from the directory
images = glob.glob('./imagesOneCam/*.png')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    if ret:
        objpoints.append(objp)
        # Refine pixel coordinates for given 2D points
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Calibration', img)
        cv2.waitKey(0)

cv2.destroyAllWindows()

# Perform camera calibration
h, w = gray.shape[:2]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print camera calibration results
print("Camera matrix:")
print(mtx)
print("Distortion coefficients:")
print(dist)

# Calculate and display distances for each image
for i, tvec in enumerate(tvecs):
    x, y, z = tvec.flatten()
    distance = np.sqrt(x**2 + y**2 + z**2)  # Euclidean distance
    print(f"\nCheckerboard position for image {i + 1}:")
    print(f"X: {x:.2f} cm, Y: {y:.2f} cm, Z: {z:.2f} cm")
    print(f"Euclidean Distance: {distance:.2f} cm")
