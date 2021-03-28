# import the necessary packages
from imagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import canny_edge_detector as ced 
import utils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# img = utils.load_data(args["image"])
# detector = ced.cannyEdgeDetector(img, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
# img_final = detector.detect()
# utils.visualize(img_final, 'gray')

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
# image = cv2.imread(args["image"])
# resized = imutils.resize(image, width=600)
# image = imutils.resize(image, width=600)
# ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
image = cv2.imread(args["image"])
# resized = imutils.resize(image, width=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
detector = ced.cannyEdgeDetector(thresh, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
img_final = detector.detect()
utils.visualize(img_final, 'gray')
gray = cv2.imread("output.png")
thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow("Resized", resized)
cv2.imshow("gray", gray)
# cv2.imshow("blur", blurred)
cv2.imshow("thresh", thresh)

# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()

# loop over the contours
for c in cnts:
    area = cv2.contourArea(c)
    if area > image.shape[0]:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]) )
        cY = int((M["m01"] / M["m00"]) )
        shape = sd.detect(c)

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        # c *= ratio
        c = c.astype("int")
        if shape != "unidentified":
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 000), 2)

# show the output image
cv2.imshow("Image", image)
# cv2.imshow("Image", img)
cv2.waitKey(0)