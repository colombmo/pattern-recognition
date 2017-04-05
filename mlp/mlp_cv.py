from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import numpy as np

f = open("reduced.csv") #open("train.csv")
saveFile = open("results.txt", "w")

data = np.loadtxt(fname = f, delimiter = ",")
X = data[:, 1:] 	 							# select features
y = data[:, 0]   								# select column 0, the digit represented by the features

min_max_scaler = preprocessing.MinMaxScaler()
sc_X = min_max_scaler.fit_transform(X)

for neurons in range(10,300,20):
	saveFile.write("\n%d" % neurons)
	for c in np.arange(0.001,0.011,0.003):
		max = 0
		for i in range(0,5):
			mlp = MLPClassifier(hidden_layer_sizes=(neurons), learning_rate_init = c)
			scores = cross_val_score(mlp, X, y, cv=5)
			if scores.mean()>max:
				max = scores.mean()
		saveFile.write("\t%.4f" % max)
		print("ok %d" % neurons)
saveFile.close()