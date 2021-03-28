# import the necessary packages
import cv2
class ShapeDetector:
	def __init__(self):
		pass

	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04 * peri, True)

		k = cv2.isContourConvex(c)
		# shape = str(len(approx))

		# if the shape has 4 vertices rectangle
		if len(approx) == 4:
			(x, y, w, h) = cv2.boundingRect(approx)
			if  peri / (float)(w + h) >= 1.9 and peri / (float)(w + h) <= 2.1:
				shape=  "button"
		# otherwise, we assume the shape is a circle
		# elif len(approx) % 2 == 0:
		# # else:
		# 	shape = "circle"
		# return the name of the shape
		return shape 
		# return str(len(approx))#