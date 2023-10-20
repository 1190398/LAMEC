'''
Sample Command:-
python detect_aruco_images.py --image Images/test_image_1.png
'''
import numpy as np
import argparse
import cv2

outputSize = 800  #change if needed to the need of the color analysis

# remove argument parser when dealing with camera input
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image containing ArUCo tag")
args = vars(ap.parse_args())

# load images
image = cv2.imread(args["image"])

# get height and width in pixels
height, width, _ = image.shape

# ArUco marker detection
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()
corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

cornerCoordinates = []

# calculate the markers center
for i in range(len(corners)):
    cornerCoordinates.append([
		int(np.mean(corners[i][0][:, 0])),
        int(np.mean(corners[i][0][:, 1]))
	])
    
# init distortion variables
pts1=np.float32(cornerCoordinates)
pts2=np.float32([[0,0],[width,0],[0,height],[width,height]])

# distort image to square
matrix=cv2.getPerspectiveTransform(pts1,pts2)
output=cv2.warpPerspective(image,matrix,(width,height))

# resize to 600*600 for square presentation
output = cv2.resize(output, (outputSize, outputSize), interpolation=cv2.INTER_CUBIC)

# remove when developing color analysis
cv2.imshow("Image", output)

# remove when developing color analysis
cv2.imwrite("output_sample.png",output)


cv2.waitKey(0)
