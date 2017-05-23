import numpy as np
from scipy.spatial.distance import euclidean

#from dtw import dtw
from fastdtw import fastdtw
import datetime

from multiprocessing import Pool
from functools import partial

# Compute dtw between two features vectors sets
def f(trainFeat, testFeat):
		dist, path = fastdtw(testFeat, trainFeat, dist=euclidean)
		return dist

# Read feature vectors from .txt
# Fill dictionary of couples id - feature vectors
if __name__ == '__main__':
	recognized = {}
	features = {}
	for type in ["train","test"]:
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
		recognized[key] = {}
		tempFeatures = []
		count = 0
		res = {}
		
		# Select all feature vectors from training set that correspond to the desired keyword
		
		for label in features["train"]:
			if transcriptions[label] == key:
				tempFeatures.append(features["train"][label])
		
		'''		
		for label in features["test"]:
			if transcriptions[label] == key:
				count = count +1
		'''
		
		# Compute distances between keyword's feature vectors and each feature vector in training set			
		if np.shape(tempFeatures)[0] > 0: #and count > 0:
			# Use multithreading to speed up everything
			p = Pool(np.shape(tempFeatures)[0])
			for label in features["test"]:
				testFeat = features["test"][label]
				res[label] = np.mean(p.map(partial(f, testFeat), tempFeatures))
			p.close()	
			# Sort elements by increasing distance
			results = sorted(res, key=res.get, reverse = False)
			
			for r in results:
				recognized[key][r] = res[r];
				
			# Compute average precision
			'''
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
			
			mean_precisions.append(np.mean(precisions))
			print("Key: "+key+", avg_precision: "+str(np.mean(precisions)))
			'''	
		
	#print("Average mean precision: "+str(np.mean(mean_precisions)))
	print("End: "+str(datetime.datetime.now().time()))
	
	# Save results to .txt file
	f1=open("results.txt", "w")
	tot = ""
	for key in recognized:
		s = key+", "
		for id in recognized[key]:
			s += id+", "+str(recognized[key][id])+", "
		s = s[:-2]
		tot+=s+"\n"
	f1.write(tot)