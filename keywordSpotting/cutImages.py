import cv2
import numpy as np
import math
import os
from xml.dom import minidom

# Split data in training and validation data
for type in ["train", "valid"]:
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
			mask = cv2.bitwise_not(mask[minY:maxY, minX:maxX])
			
			# Mask image
			masked_image = cv2.bitwise_or(img, mask)
			
			# Image binarization, with Otsu's thresholding
			r, image = cv2.threshold(masked_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			
			'''
			## In reality we don't really care about skew, since it is handled well enough by the already done separation of the words
			
			# Find skew with lower contour pixel regression
			lower_pixels = {"x":[], "y":[]}
			for x in range(0,image.shape[1]):
				try:
					lower_pixels["y"].append(np.amax(np.where(image[:,x] < 255)))
					lower_pixels["x"].append(x)
				except:
					continue

			# Slope of linear regression on the lower contour points
			m, b = np.polyfit(lower_pixels["x"], lower_pixels["y"], 1)
			# Skew
			angle = -np.arctan(m);
			print(angle)
			# rotate the image to deskew it
			(h, w) = image.shape[:2]
			center = (w // 2, h // 2)
			M = cv2.getRotationMatrix2D(center, math.degrees(angle), 1.0)
			rotated = cv2.warpAffine(image, M, (w, h),
				flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
			'''
			# Save preprocessed image
			cv2.imwrite("images/"+type+"/"+ids[i]+".jpg",image)