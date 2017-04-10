from sklearn.model_selection import GridSearchCV
from sklearn import svm, metrics, utils
import matplotlib.pyplot as plt
import numpy as np
import time

print('Start time:' + time.ctime())

trainingSet = np.genfromtxt('train.csv', delimiter=",")
testSet = np.genfromtxt('test.csv', delimiter=",")

print('Data loaded:' + time.ctime())

# Shuffle data
trainingSet = utils.shuffle(trainingSet)
testSet = utils.shuffle(testSet)

trainingLabels = trainingSet[:,0]
trainingData = trainingSet[:,1:]

testLabels = testSet[:,0]
testData = testSet[:,1:]

row, col = trainingSet.shape

# Plot some original data
dataAndLabels = list(zip(trainingSet, trainingLabels))
for index, (image, label) in enumerate(dataAndLabels[:6]):
    plt.subplot(2, 6, index + 1)
    plt.axis('off')
    plt.imshow(trainingSet[index,1:col].reshape((28,28)), cmap=plt.cm.Greys_r, interpolation='nearest')
    plt.title('Training: %i' % label, fontsize=6)

#classifier = svm.LinearSVC()
classifier = svm.LinearSVC(C=0.00001)

# Grid search
# params = {'C': [0.00001, 0.0001, 0.001, 1]}
# gridSearch = GridSearchCV(classifier, params)
# gridSearch.fit(trainingData,trainingLabels)
# predictedLabels = gridSearch.predict(testData)

classifier.fit(trainingData, trainingLabels)
predictedLabels = classifier.predict(testData)

print('The end:' + time.ctime())

# print ("C = ", gridSearch.best_estimator_.C)

print("Classification report %s:n%sn"
      % (classifier, metrics.classification_report(testLabels, predictedLabels)))
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