## How to execute ##
Go to MNIST/ and then
  python mlp.py
  
At the end of execution, the results can be found in the file results.txt

## Parameters selection ##
During cross validation, we did a grid search for several possible parameters of our MLP.
This part has been commented out for performance reasons, but it's still in the code to show how we did it.

The parameters that gave us the best results during cross validation, and that we picked for the MLP classifier, are the following:
  solver='sgd', 
  activation='relu',  
  learning rate=0.3,
  neurons = 100,
  alpha=0.01
  max_iter=200

With those parameters we get an accuracy >97% on the test set.
