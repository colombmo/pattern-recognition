from sklearn.neural_network import MLPClassifier
import numpy as np

ft = open("train.csv")
ftst = open("test.csv")

train = np.loadtxt(fname = ft, delimiter = ",")
test = np.loadtxt(fname = ftst, delimiter = ",")

X_train = train[:, 1:] 
y_train = train[:, 0]   

X_test = test[:, 1:]
y_test = test[:, 0]

print("Files loaded")

mlp = MLPClassifier(hidden_layer_sizes=(270), learning_rate_init = 0.004)
mlp.fit(X_train, y_train)
print("MLP trained")
score = mlp.score(X_test, y_test)

print("MLP accuracy = %.4f" % score)