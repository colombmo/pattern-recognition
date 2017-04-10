import time
import numpy as np

import matplotlib.pyplot as plt
from sklearn import svm, metrics, utils
from numpy import genfromtxt
from skimage.feature import hog
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

print('Start time:' + time.ctime())

trainingSet = genfromtxt('train.csv', delimiter=",")
testSet = genfromtxt('test.csv', delimiter=",")

print('Data loaded:' + time.ctime())

# Shuffle data
trainingSet = utils.shuffle(trainingSet)
testSet = utils.shuffle(testSet)

trainingLabels = trainingSet[:,0]
trainingData = trainingSet[:,1:]

testLabels = testSet[:,0]
testData = testSet[:,1:]

#Scaling data in interval [0,1]
min_max_scaler = preprocessing.MinMaxScaler()
sc_trainingData = min_max_scaler.fit_transform(trainingData)
sc_testData = min_max_scaler.transform(testData)


row, col = trainingSet.shape
# Plot some original data
dataAndLabels = list(zip(trainingSet, trainingLabels))
for index, (image, label) in enumerate(dataAndLabels[:6]):
    plt.subplot(2, 6, index + 1)
    plt.axis('off')
    plt.imshow(trainingSet[index,1:col].reshape((28,28)), cmap=plt.cm.Greys_r, interpolation='nearest')
    plt.title('Training: %i' % label, fontsize=6)

# For a faster performance we can use HoG features but we will
# and get a solid accuracy but around 5 percent lower than with the raw features

# # HoG features for training data
# hogFeaturesList = []
# for feature in trainingData:
#     fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
#     hogFeaturesList.append(fd)
# hogFeatures = np.array(hogFeaturesList, 'float64')


# # HoG features for test data
# hog_ft = []
# for feature in testData:
#     ft = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
#     hog_ft.append(ft)
# hogTestFeatures = np.array(hog_ft, 'float64')

'''
We performed grid search to find the optimal values for parameters C and gamma
from the list that we tested(C=5, gamma=0.05 were the best).
Grid search is very time consuming so we will comment it out
'''

svc = svm.SVC(kernel='rbf', cache_size=1000, C=5, gamma=0.05)

svc.fit(sc_trainingData, trainingLabels)

predictedLabels = svc.predict(sc_testData)


#params = {'C': [0.01, 0.1, 1, 5], 'gamma': [0.001, 0.01, 0.05, 0.1]}
#gridSearch =  GridSearchCV(svc, params)
#gridSearch.fit(hogFeatures,trainingLabels)
#predictedLabels = gridSearch.predict(hogTestFeatures)


print('The end:' + time.ctime())

#print "Best params:", gridSearch.best_params_

print("Classification report %s:n%sn" % (svc, metrics.classification_report(testLabels, predictedLabels)))

# Check how many times a digit was recognized as a certain digit.
print("Confusion matrix:\n %s" % metrics.confusion_matrix(testLabels, predictedLabels))

print("Accuracy:\n %s" % metrics.accuracy_score(testLabels, predictedLabels))

# Plot predicted data
dataAndPredictions = list(zip(testSet, predictedLabels))
for index, (image, prediction) in enumerate(dataAndPredictions[:6]):
    plt.subplot(2, 6, index + 7)
    plt.axis('off')
    plt.imshow(testSet[index,1:col].reshape((28,28)), cmap=plt.cm.Greys_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction, fontsize=6)
plt.show()