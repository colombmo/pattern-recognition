from sklearn import svm, metrics
import time
import numpy as np

from sklearn.model_selection import GridSearchCV

print('Start time:' + time.ctime())

trainingSet = np.genfromtxt('train.csv', delimiter=",")
testSet = np.genfromtxt('test.csv', delimiter=",")

print('Data loaded:' + time.ctime())

trainingLabels = trainingSet[:,0]
trainingData = trainingSet[:,1:]

testLabels = testSet[:,0]
testData = testSet[:,1:]

classifier = svm.LinearSVC()

params = {'C': [0.0001, 0.001, 0.1, 1]}

gridSearch = GridSearchCV(classifier, params)

gridSearch.fit(trainingData,trainingLabels)

predictedLabels = gridSearch.predict(testData)

#classifier.fit(trainingData, trainingLabels)

#predictedLabels = classifier.predict(testData)

print('The end:' + time.ctime())

print "C = ", gridSearch.best_estimator_.C

print("Classification report %s:n%sn"
      % (classifier, metrics.classification_report(testLabels, predictedLabels)))

