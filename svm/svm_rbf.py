import time
import numpy as np

from sklearn import svm, metrics
from numpy import genfromtxt
from skimage.feature import hog
from sklearn.model_selection import GridSearchCV


print('Start time:' + time.ctime())

trainingSet = genfromtxt('train.csv', delimiter=",")
testSet = genfromtxt('test.csv', delimiter=",")

print('Data loaded:' + time.ctime())

trainingLabels = trainingSet[:,0]
trainingData = trainingSet[:,1:]

testLabels = testSet[:,0]
testData = testSet[:,1:]

# HoG features for training data
hogFeaturesList = []
for feature in trainingData:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    hogFeaturesList.append(fd)
hogFeatures = np.array(hogFeaturesList, 'float64')

# HoG features for test data
hog_ft = []
for feature in testData:
    ft = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    hog_ft.append(ft)
hogTestFeatures = np.array(hog_ft, 'float64')


'''
We performed grid search to find the optimal values for parameters C and gamma
from the list that we tested(C=5, gamma=0.05 were the best).
Grid search is very time consuming so we will comment it out
'''

svc = svm.SVC(kernel='rbf', cache_size=800, C=5, gamma=0.05)

svc.fit(hogFeatures, trainingLabels)

predictedLabels = svc.predict(hogTestFeatures)


#params = {'C': [0.01, 0.1, 1, 5], 'gamma': [0.001, 0.01, 0.05, 0.1]}
#gridSearch =  GridSearchCV(svc, params)
#gridSearch.fit(hogFeatures,trainingLabels)
#predictedLabels = gridSearch.predict(hogTestFeatures)


print('The end:' + time.ctime())

#print "Best params:", gridSearch.best_params_

print("Classification report %s:n%sn" % (svc, metrics.classification_report(testLabels, predictedLabels)))
