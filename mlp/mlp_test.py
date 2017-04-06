from sklearn.neural_network import MLPClassifier
import numpy as np

from sklearn import preprocessing

ft = open("train.csv")
ftst = open("test.csv")

train = np.loadtxt(fname = ft, delimiter = ",")
test = np.loadtxt(fname = ftst, delimiter = ",")

X_train = train[:, 1:] 
y_train = train[:, 0]   

X_test = test[:, 1:]
y_test = test[:, 0]

min_max_scaler = preprocessing.MinMaxScaler()
sc_X_train = min_max_scaler.fit_transform(X_train)
sc_X_test = min_max_scaler.fit_transform(X_test)

print("Files loaded")

mlp = MLPClassifier(hidden_layer_sizes=(230), learning_rate_init = 0.004)
mlp.fit(sc_X_train, y_train)
print("MLP trained")
score = mlp.score(sc_X_test, y_test)

print("MLP accuracy = %.4f" % score)

for i in range(1, 22, 1):
	max = 0
	mean = 0
	for j in range(0,10):
		mlp = MLPClassifier(hidden_layer_sizes=(230), learning_rate_init = 0.004, max_iter = i)
		mlp.fit(sc_X_train, y_train)
		#print("MLP trained")
		score = mlp.score(sc_X_test, y_test)
		mean = (mean*j+score)/(j+1)
		if max < score:
			max = score;
	print("%d\t%.4f" % (i, max))
	print("%d\t%.4f" % (i, mean))
		#print("MLP accuracy = %.4f" % score)