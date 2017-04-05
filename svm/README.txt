Linear kernel, 5-fold cross validation (on subset of 1000 elements on the training set, for time reduction purposes)

C		score
2^-20   0.5414
2^-19   0.5563
2^-18   0.5862
2^-17   0.6202
2^-16   0.6540
2^-15   0.7019
2^-14   0.7318
2^-13   0.7568
2^-12   0.7728
2^-11   0.7849
2^-10   0.8058
2^-9    0.8248
2^-8    0.8439
2^-7    0.8559
2^-6    0.8568
2^-5    0.8548
2^-4    0.8499
2^-3    0.8429
2^-2    0.8351
2^-1    0.8331
2^0     0.8301
2^1     0.8312
2^2     0.8252
2^3     0.8233

=> So here we pick C = 2^-6, since it is the one which gives the best accuracy during our cross-validation.
	
With this parameter, we train the classifier on the complete training set and we test it on the test set, getting an accuracy:

	Accuracy_linear = 0.9101
	
__________________________________________________________________________________________________________________________________________________________
Polynomial kernel, 5-fold cross validation (on subset of 1000 elements on the training set, for time reduction purposes)

	gamma	2^-15	2^-10	2^-5	2^0
C
2^-10		0.1120	0.1120	0.1700	0.8637
2^-8		0.1120	0.1120	0.5169	0.8637
2^-6		0.1120	0.1120	0.7516	0.8637
2^-4		0.1120	0.1120	0.8386	0.8637
2^-2		0.1120	0.1120	0.8507	0.8637
2^0			0.1120	0.1120	0.8637	0.8637
2^2			0.1120	0.1120	0.8637	0.8637
2^4			0.1120	0.1240	0.8637	0.8637
2^6			0.1120	0.2860	0.8637	0.8637
2^8			0.1120	0.6409	0.8637	0.8637
2^10		0.1120	0.8038	0.8637	0.8637
2^12		0.1120	0.8457	0.8637	0.8637
2^14		0.1120	0.8576	0.8637	0.8637


=> Here we can see that the best score we can get in this cross-validation with any C and gamma combination is 0.8637.
	So we can pick any combination that gives this as a result. We therefore randomly pick C = 2^12, gamma = 2^0.
	
With those parameters, we train the classifier on the complete training set and we test it on the test set, getting an accuracy:

	Accuracy_polynomial = 0.9726