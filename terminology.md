# batch_size
### Number of observations that are passed through the network at once during an epoch. 
- All training observations will ultimately be passed through during the epoch. batch_size determines the number that have their gradients combined before backpropagation.
- <batch_size> observations are propagated forward, and then backpropagation updates the weights. Then the next <batch_size> observations are processed. 
- A batch_size of 1 is also called Stochastic Gradient Descent. 
  - One observation is passed through, the error is calculated, the gradient for each node is calculated, the parameters are updated, and then the cycle repeats
  - With Stochasic Gradient Descent, the gradient is calculated very accurately, but it takes a long time to train. 
- A batch_size equal to the number of training observations is called Batch Gradient Descent
  - All observations are still passed through one at a time, and their gradients are calculated. The gradients for all observations are then summed, and the parameters are updated using this summed gradient.
  - There is therefore only 1 forward pass/backpropagation cycle.
  - With BGD, all training data must be stored in memory
- A batch_size between 1 and the total number of training observations is called Mini Batch Gradient Descent
  - <batch_size> obervations (sequentially selected by default) are selected from the training data. The same procedure as Batch Gradient Descent is then performed. Then the next batch of samples are selelcted, and the cycle continues
#### Links
- https://ai.stackexchange.com/a/20380
- https://towardsdatascience.com/batch-mini-batch-stochastic-gradient-descent-7a62ecba642a

# Epoch
### A full iteration over the training dataset during model training.
- Usually, multiple epochs are used and their improvement (or worsening) at accuracy is tracked. 

# Hyperparameters
### Model configuration specs, e.g. learning_rate, batch_size

# Learning rate
### Amount the gradient is multiplied by during backpropagation to update a weight.
- Determines the size of the steps taken during gradient decent
- Smaller learning rates mean smaller steps taken during gradient descent. This can lead to being trapped in local minimums, and requires more epochs to hit peak accuracy (possibly more than are necessary). 
- Larger learning rates can lead to divergence as the steps taken move wildly along the slopes
#### Links
- https://www.jeremyjordan.me/nn-learning-rate/

# Parameters
### Weights and biases for each node

# Validation data
### A subset of the overall dataset that is used during model training to assist with hyperparameter tuning. 
- The model is not trained on this data. Rather, this data is used during each epoch to test the model's accuracy after gradient descent is complete.
- If accuracy on the validation set seems low, divergent, slow, etc, hyperparameters and/or preprocesing should be done to improve the model.
#### Links
-https://stackoverflow.com/a/46308466/10199577

## Generally useful links
- https://developers.google.com/machine-learning/glossary