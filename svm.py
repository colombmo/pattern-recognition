from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

f = open("reduced.csv") #open("train.csv")
data = np.loadtxt(fname = f, delimiter = ",")
X = data[:, 1:] 	 							# select features
y = data[:, 0]   								# select column 0, the digit represented by the features
print("File loaded")

#kf = KFold(n_splits=10)							# 10-fold cross validation
ker = "linear"	
for c in range(-23,-19,1):
	svc = svm.SVC(kernel = ker, C=2**c)
	scores = cross_val_score(svc, X, y, cv=5)

	print("%s kernel score, C = 2^%d: %.4f +- %.4f" % (ker, c, scores.mean(), scores.std()*2))

ker = "poly"	

for c in range(-10,10,2):
	for g in range(-25,-20,1):
		svc = svm.SVC(kernel = ker, C=2**c, gamma = 2**g)
		scores = cross_val_score(svc, X, y, cv=5)

		print("%s kernel score, C = 2^%d, gamma = 2^%d: %.4f +- %.4f" % (ker, c, g, scores.mean(), scores.std()*2))
		
'''
for i in range(-5, 3, 2):
	totalScore = 0

	for train, test in kf.split(X):
		X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
													# Get features and represented digits from the data
		linear_svc = svm.SVC(kernel="linear", C = 2**i)# Build an SVM with linear kernel
		linear_svc.fit(X_train, y_train)			# Train the SVM
		score = linear_svc.score(X_test, y_test)	# Score of the SVM on the validation data
		#print("Score linear: %f" % score)
		
		totalScore += score

	print("Mean score linear kernel, C = 2^(%d): %f" % (i , (totalScore/10)))
'''
'''
for c in range(-5,15,2):
	for g in range(-13,3,2):
		for train, test in kf.split(X):
			totalScore = 0
			
			X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
														# Get features and represented digits from the data
			rbf_svc = svm.SVC(kernel="rbf", C = 2**c, gamma = 2**g)
														# Build an SVM with rbf kernel
			rbf_svc.fit(X_train, y_train)				# Train the SVM
			score = rbf_svc.score(X_test, y_test)		# Score of the SVM on the validation data
			#print("Score linear: %f" % score)
			
			totalScore += score	

		print("Mean score rbf kernel, C = 2^%d, gamma = 2^%d : %f" % (c, g, (totalScore/10)))
'''