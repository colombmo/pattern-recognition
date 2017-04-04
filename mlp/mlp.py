from sklearn.model_selection import GridSearchCV
from sklearn import svm, metrics, utils, neural_network
import matplotlib.pyplot as plt
import numpy as np
import time

print('Start time:' + time.ctime())

trainingSet = np.genfromtxt('../svm/train.csv', delimiter=",")
testSet = np.genfromtxt('../svm/test.csv', delimiter=",")

print('Data loaded:' + time.ctime())

# Suffle data
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

print('Enter MLP...')

classifier = neural_network.MLPClassifier(hidden_layer_sizes=(50), max_iter=500, alpha=0.0001, solver='sgd', tol=0.0001, random_state=1, learning_rate='constant', learning_rate_init=0.001)
classifier.fit(trainingData, trainingLabels)

predictedLabels = classifier.predict(testData)

print('The end:' + time.ctime())

print("Classification report %s:n%sn" % (classifier, metrics.classification_report(testLabels, predictedLabels)))
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