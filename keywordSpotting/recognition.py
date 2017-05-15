import numpy as np
from scipy.spatial.distance import euclidean

#from dtw import dtw
from fastdtw import fastdtw
import datetime

# Read feature vectors from .txt
# Fill dictionary of couples id - feature vectors
features = {}
for type in ["train","valid"]:
	features[type] = {}
	with open("features/features_"+type+".txt", "r") as myfile:
		lines = myfile.readlines()
		for l in lines:
			a = l.replace("\n","").split(", ")
			feats = []
			for el in range(1,len(a),4):
				feats.append([float(a[el]),float(a[el+1]),float(a[el+2]),float(a[el+3])])
			features[type][a[0]] = np.array(feats)

# Load transcriptions
transcriptions = {}
with open("ground-truth/transcription.txt", "r") as myfile:
		lines = myfile.readlines()
		for l in lines:
			a = l.replace("\n","").split(" ")
			transcriptions[a[0]] = a[1]
			
# Load list of keywords
with open("task/keywords.txt", "r") as myfile:
	keywords = [l.replace("\n","") for l in myfile.readlines()]

print("Start: "+str(datetime.datetime.now().time()))
	
# For each keyword, try to recognize it in the test set
mean_precisions = []
for key in keywords:
	tempFeatures = []

	# Select all feature vectors from training set that correspond to the desired keyword
	for i in features["train"]:
		if transcriptions[i] == key:
			tempFeatures.append(features["train"][i])
	
	# Compute distances between keyword's feature vectors and each feature vector in training set	
	count = 0
	res = {}
	for label in features["valid"]:
		if transcriptions[label] == key:
			count = count +1
	
	if count > 0:
		for label in features["valid"]:
			dists = []
			for trainFeat in tempFeatures:
				dist, path = fastdtw(features["valid"][label], trainFeat, dist=euclidean)
				dists.append(dist)
			res[label] = np.mean(dists)
		
		# Sort elements by increasing distance
		results = sorted(res, key=res.get, reverse = False)
		
		#pr = [transcriptions[r] for r in results[:10]]
		#print(pr)
		
		# Compute average precision
		precisions = []
		tp = 0
		fp = 0

		for r in results:
			if tp == count:
				break
			if transcriptions[r] == key:
				tp = tp+1
			else:
				fp=fp+1
			precisions.append(tp/(tp+fp))

		if np.mean(precisions)>0:
			mean_precisions.append(np.mean(precisions))

		#print("Key: "+key+", avg_precision: "+str(np.mean(precisions)))

#print(mean_precisions)
print("Using max of distances")
print("Average mean precision: "+str(np.mean(mean_precisions)))
print("End: "+str(datetime.datetime.now().time()))