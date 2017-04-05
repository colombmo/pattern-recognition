from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn import preprocessing
import numpy as np

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

linear_svc = LinearSVC(C=2**(-6))
linear_svc.fit(sc_X_train, y_train)
print("Linear svm trained")
score = linear_svc.score(sc_X_test, y_test)

print("Linear kernel score: %.4f" % score)

poly_svc = svm.SVC(kernel = "poly", C=2**12, gamma = 2**0)
poly_svc.fit(sc_X_train, y_train)
print("Polynomial svm trained")
score = poly_svc.score(sc_X_test, y_test)

print("Polynomial kernel score: %.4f" % score)
