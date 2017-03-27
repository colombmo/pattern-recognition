from sklearn import svm
from sklearn.model_selection import KFold
import numpy as np

f = open("reduced.csv")#open("train.csv")
data = np.loadtxt(fname = f, delimiter = ",")
X = data[:, 1:] 	 							# select features
y = data[:, 0]   								# select column 0, the digit represented by the features
print("File loaded")

kf = KFold(n_splits=10)							# 10-fold cross validation
totalScore = 0

for train, test in kf.split(X):
	X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
												# Get features and represented digits from the data
	linear_svc = svm.SVC(kernel="linear")		# Build an SVM with linear kernel
	linear_svc.fit(X_train, y_train)			# Train the SVM
	score = linear_svc.score(X_test, y_test)	# Score of the SVM on the validation data
	print("Score linear: %f" % score)
	
	totalScore += score
	
print("Mean score linear kernel: %f" % totalScore/10)