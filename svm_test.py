from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
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

linear_svc = svm.SVC(kernel = "linear", C=2**(-21))
linear_svc.fit(X_train, y_train)
print("Linear svm trained")
score = linear_svc.score(X_test, y_test)

print("Linear kernel score: %.4f" % score)

poly_svc = svm.SVC(kernel = "poly", C=2**6, gamma = 2**(-21))
poly_svc.fit(X_train, y_train)
print("Polynomial svm trained")
score = poly_svc.score(X_test, y_test)

print("Polynomial kernel score: %.4f" % score)
