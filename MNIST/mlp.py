from sklearn.model_selection import GridSearchCV
from sklearn import svm, metrics, utils, preprocessing
import numpy as np
import time

from sklearn.neural_network import MLPClassifier

print('Start time:' + time.ctime())

trainingSet = np.genfromtxt('train.csv', delimiter=",")
testSet = np.genfromtxt('test.csv', delimiter=",")

print('Data loaded:' + time.ctime())

trainingLabels = trainingSet[:, 0]
trainingData = trainingSet[:, 1:]

testData = testSet

min_max_scaler = preprocessing.MinMaxScaler()
sc_trainingData = min_max_scaler.fit_transform(trainingData)
sc_testData = min_max_scaler.transform(testData)


clf = MLPClassifier(solver='sgd', activation='relu', learning_rate='constant', learning_rate_init=0.3,
                    hidden_layer_sizes=(100,), alpha=0.01)

# Optimizations 1,2,3
# To apply cross validation uncomment the code below and remove those
# parameters from mlpclassifier above

# lr_range = np.linspace(0.1, 1, 10)
# neurons_range = np.linspace(10, 100, 10)
# iter_range = np.linspace(100, 200, 3): 100,150,200 iterations

# Grid search
# params = {'learning_rate_init': lr_range, 'hiden_layer_sizes':[(10,),(25,),(50,),(75,),(100,)]}
# params = {'solver':['sgd','adam'],'activation':['identity', 'logistic', 'tanh', 'relu']}
# params = {'max_iter': iter_range}
# params = {'alpha':[1e-2, 1e-3, 1e-4, 1e-5]}
# gridSearch = GridSearchCV(clf, params, n_jobs=-1)
# gridSearch.fit(sc_trainingData, trainingLabels)
# predictedLabels = gridSearch.predict(sc_testData)

clf.fit(sc_trainingData, trainingLabels)
predictedLabels = clf.predict(sc_testData)

# Save results to .txt file
f1=open("results.txt", "w")
tot = ""
for id, label in enumerate(predictedLabels):
	s = str(id)+", "+str(int(label))
	tot+=s+"\n"
f1.write(tot)

print('The end:' + time.ctime())

#print 'Best params:', gridSearch.best_params_

#print("Classification report %s:n%sn"
#      % (clf, metrics.classification_report(testLabels, predictedLabels)))
# Check how many times a digit was recognized as a certain digit.
#print("Confusion matrix:\n %s" % metrics.confusion_matrix(testLabels, predictedLabels))
#print("Accuracy:\n %s" % metrics.accuracy_score(testLabels, predictedLabels))
