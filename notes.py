### Neural Net comments ###

# Tweak training/test/validation split
# Shuffling training data
#   - If shuffling the data causes big changes in accuracy, possibly too little data
#   - Consider K-fold validation
# If two observations are identical
#   - dont include one in training and one in validation, bc it's not really a validation then
#
# Dropout- randomly setting some of the output vector of a layer to be 0 when training
#   - can reduce overfitting
#   - only done during training
#
# Data augmentation/synthetic data
#   - both are about making more data to train the model
#   - data augmentation: altering training data (i.e. rotating an image)
#   - synthetic data: creating training data using an algorithm (can be hybrid and contain
#     some real data elements)
#   - could this ever be useful for our problems?
#
# Interpreting the neurons
#   - is it possible to create observations that maximize the response of a neuron in our use case the way
#     it's done with filters for images?
#   - For MNIST, visualize the neurons and then increase the number of layers and try again
#
# Ensemble
#   - Use multiple models (deep learning, decision trees) and produce a weighted mean of the weights


### AWS potential services to use ###
# S3
#   - storing models
#   - storing prepped data? depends on size, don't want to have to pay

