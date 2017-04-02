Linear kernel, 5-fold cross validation (on subset of 1000 elements on the training set, for time reduction purposes)

C					score
2^-28		 		0.1190
2^-26				0.6869
2^-24				0.8426
2^-23				0.8567
2^-22				0.8716
2^-21				0.8767
2^-20				0.8759
2^-18				0.8639

=> So here we pick C = 2^-21, since it is the one which gives the best accuracy during our cross-validation. (Also bigger and smaller Cs where tried, but
	for all C <= 2^-28, we had score = 0.1190; and for all C >= 2^-18, score = 0.8639).
	
With this parameter, we train the classifier on the complete training set and we test it on the test set, getting an accuracy:

	Accuracy_linear = 0.9417
	
__________________________________________________________________________________________________________________________________________________________
Polynomial kernel, 5-fold cross validation

	gamma	2^-25	2^-24	2^-23	2^-22	2^-21
C
2^-10		0.1120	0.1120	0.1120	0.1120	0.1680
2^-8		0.1120	0.1120	0.1120	0.1210	0.5049
2^-6		0.1120	0.1120	0.1120	0.2850	0.7486
2^-4		0.1120	0.1120	0.1680	0.6370	0.8386
2^-2		0.1120	0.1210	0.5049	0.7998	0.8497
2^0			0.1120	0.2850	0.7486	0.8457	0.8637
2^2			0.1680	0.6370	0.8386	0.8567	0.8637
2^4			0.5049	0.7998	0.8497	0.8637	0.8637
2^6			0.7486	0.8457	0.8637	0.8637	0.8637
2^8			0.8386	0.8567	0.8637	0.8637	0.8637


=> Here we can see that the best score we can get in this cross-validation with any C and gamma combination is 0.8637.
	So we can pick any combination that gives this as a result. We therefore randomly pick C = 2^6, gamma = 2^-21.
	
With those parameters, we train the classifier on the complete training set and we test it on the test set, getting an accuracy:

	Accuracy_polynomial = 0.9726