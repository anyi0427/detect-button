import utils
import numpy as np
import canny_edge_detector as ced
import cv2, imutils
from ShapeDetector import ShapeDetector
file_name = "imgs/6.png"
img = utils.load_data(file_name)
final_img = cv2.imread(file_name)
utils.visualize(img, 'gray')

detector = ced.cannyEdgeDetector(img, sigma=1.2, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=80)

image = detector.detect()
utils.visualize(image, 'gray')

# s = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# s = cv2.Laplacian(image , cv2.CV_16S, ksize=3)
# s = cv2.convertScaleAbs(s)
image = image.astype(np.uint8)

resized = imutils.resize(image, width=600)
final_img = imutils.resize(final_img, width=600)
# ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(resized, 10, 255, cv2.THRESH_BINARY)[1]

# cv2.imshow("Resized", resized)
# cv2.imshow("gray", gray)
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
    if area > final_img.shape[0] and  area < image.shape[0] * image.shape[0]/8 :
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]) )
        cY = int((M["m01"] / M["m00"]) )
        shape = sd.detect(c)

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c = c.astype("int")
        if shape != "unidentified":
            cv2.drawContours(final_img, [c], -1, (0, 255, 0), 2)
            cv2.putText(final_img, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255), 2)

# show the output image
cv2.imshow("Image", final_img)
# cv2.imshow("Image", img)
cv2.waitKey(0)