# Imports
import os
import numpy as np
import cv2
'''
	Process the data, by creating feature vectors from the images.
	1. Scale images
	2. Generate feature vectors for each images (only 4 features left after features selection):
		- Upper contour
		- Lower contour
		- #b/w transitions
		- width/height ratio
	3. Normalize vectors
'''

width = 50
height = 100
data = {}
values = {}

# Fill dictionary of couples id - word
with open("ground-truth/transcription.txt", "r") as myfile:
	lines = myfile.readlines()
	for l in lines:
		a = l.replace("\n","").split(' ')
		values[a[0]] = a[1]

for type in ["train", "test"]:
	data[type] = {}
	
	for filename in os.listdir("images/"+type):
		# Open image
		img = cv2.imread("images/"+type+"/"+filename, 0)
		# Get original width/height ratio and store it
		ratio = img.shape[1]/img.shape[0]
		# Scale image
		img = cv2.resize(img,(width, height), interpolation = cv2.INTER_CUBIC)         
		
		#Sliding window
		features = np.zeros((img.shape[1],4), dtype = np.float)
		for i in range(0,img.shape[1]):
			col = img[:,i]
			try:
				features[i][0] = np.amin(np.where(col<255))					# Upper contour
				features[i][1] = np.amax(np.where(col<255))					# Lower contour
			except:
				pass
				
			# b/w transitions
			trans = 0
			for j in range(1,img.shape[0]):
				if img[j,i]!=img[j-1,i]:
					trans = trans+1
			features[i][2] = trans
			# Width/height ratio
			features[i][3] = ratio
			
		data[type][filename.replace(".jpg","")] = features;

# Normalize using training data
t_mean = np.mean([np.mean(data["train"][n], axis=0) for n in data["train"]], axis=0)
t_sd = np.std([np.mean(data["train"][n], axis=0) for n in data["train"]], axis=0)	

# Create new folder for putting features
try:
	os.mkdir("features")
except:
	pass;

# Save features to file
for t in data:
	f1=open("features/features_"+t+".txt", "w")
	tot = ""
	for n in data[t]:
		s = n+", "
		for x in data[t][n]:
			for i,y in enumerate(x):
				s += str((y-t_mean[i])/t_sd[i])+", "
		s = s[:-2]
		tot+=s+"\n"
	f1.write(tot)