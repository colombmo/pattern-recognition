import cv2
import numpy as np
import math
import os
from xml.dom import minidom

# Split data in training and validation data
for type in ["train", "test"]:
	# Create new folder for putting single images
	try:
		os.mkdir("images/"+type)
	except:
		pass;
	# Get name of images
	with open("task/"+type+".txt", "r") as myfile:
		lines = myfile.readlines()
	for fn in lines:
		filenum = fn.replace("\n","")
		
		# original image
		originalImg = cv2.imread("images/"+filenum+".jpg", 0)
		
		# open svg file containing the contour of each word, to be used to select individual words
		with open("ground-truth/locations/"+filenum+".svg", "r") as myfile:
			data=myfile.read().replace("\n", "")

		# Read points of path from svg file
		path_strings = [path.getAttribute("d") for path in minidom.parseString(data).getElementsByTagName("path")]

		l = []
		for s in path_strings:
			sl = []
			for t in s.split():
				try:
					sl.append(float(t))
				except ValueError:
					pass
			l.append(sl)

		points = [[(sl[i-1],sl[i]) for i in range(1,len(sl),2)] for sl in l]

		# Read id of words from svg file
		ids = [path.getAttribute("id") for path in minidom.parseString(data).getElementsByTagName("path")]
		
		
		# Image binarization, with Gaussian thresholding
		originalImg = cv2.adaptiveThreshold(originalImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,99,30)
		
		for i,p in enumerate(points):
			img = originalImg
			
			# Get min and max xs and ys, to create a bounding box to crop the image
			#p = points[i]
			mins = list(map(min, zip(*p)))
			maxs = list(map(max, zip(*p)))
			minX = int(mins[0])
			minY = int(mins[1])
			maxX = int(maxs[0])
			maxY = int(maxs[1])
			
			# Create mask to only show the wanted word
			mask = np.zeros(img.shape, dtype=np.uint8)
			roi_corners = np.array([p], dtype=np.int32)
			ignore_mask_color = 255
			cv2.fillPoly(mask, roi_corners, ignore_mask_color)
			
			# Crop image and mask, to only focus on the area around the wanted text piece
			img = img[minY:maxY, minX:maxX]
			
			# Image binarization, with Otsu's thresholding
			#r, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
						
			mask = cv2.bitwise_not(mask[minY:maxY, minX:maxX])
			
			# Mask image
			image = cv2.bitwise_or(img, mask)
			
			# Cut white borders from image
			im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			
			xm = []; ym = []; xM = []; yM = []
			contours = contours[1:]
			for cnt in contours:
				x,y,w,h = cv2.boundingRect(cnt)
				xm.append(x); ym.append(y); xM.append(x+w); yM.append(y+h)
			try:
				image = image[min(ym)+1:max(yM)-1, min(xm)+1:max(xM)-1]
			except:
				pass
			
			# Save preprocessed image
			cv2.imwrite("images/"+type+"/"+ids[i]+".jpg",image)