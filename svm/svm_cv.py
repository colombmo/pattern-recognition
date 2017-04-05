from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn import preprocessing
import numpy as np

f = open("train.csv") #open("reduced.csv") #
data = np.loadtxt(fname = f, delimiter = ",")
X = data[:, 1:] 	 							# select features
y = data[:, 0]   								# select column 0, the digit represented by the features

min_max_scaler = preprocessing.MinMaxScaler()
sc_X = min_max_scaler.fit_transform(X)

print("File loaded")
print("Linear kernel:\nC\tscore")

ker = "linear"	
for c in range(-20,3,1):
	svc = LinearSVC(C=2**c)
	scores = cross_val_score(svc, sc_X, y, cv=5)

	print("2^%d\t%.4f" % (c, scores.mean()))

f = open("reduced.csv") 						# Reduced dataset (1000 elements), for timing purposes
data = np.loadtxt(fname = f, delimiter = ",")
X = data[:, 1:] 	 							# select features
y = data[:, 0]   								# select column 0, the digit represented by the features

min_max_scaler = preprocessing.MinMaxScaler()
sc_X = min_max_scaler.fit_transform(X)	

print("Polynomial kernel:")
ker = "poly"	
for c in range(-10,15,2):
	for g in range(-15,5,5):
		svc = svm.SVC(kernel = ker, C=2**c, gamma = 2**g)
		scores = cross_val_score(svc, sc_X, y, cv=5)

		print("C = 2^%d, gamma = 2^%d: %.4f +- %.4f" % (ker, c, g, scores.mean(), scores.std()*2))