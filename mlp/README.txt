During cross validation, we did a grid search for several possible parameters of our MLP.
This part has been commented out for performance reasons, but it's still in the code to show how we did it.

The parameters that gave us the best results during cross validation, and that we picked for the MLP classifier, are the following:
  solver='sgd', 
  activation='relu',  
  learning rate=0.3,
  neurons = 100,
  alpha=0.01
