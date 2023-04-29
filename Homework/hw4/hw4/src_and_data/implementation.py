import numpy as np


def counting_heuristic(x_inputs, y_outputs, feature_index, classes):
    """
    Calculate the total number of correctly classified instances for a given
    feature index, using the counting heuristic.

    :param x_inputs: numpy array of shape (n_samples, n_features), containing the input data
    :param y_outputs: numpy array of shape (n_samples,), containing the output data (class labels)
    :param feature_index: int, index of the feature to be evaluated
    :param classes: container, unique set of class labels.
           e.g. [0,1] for two classes or [0,1,2] for 3 classes
    :return: int, total number of correctly classified instances
    """
    
    n_samples, n_features = x_inputs.shape

    total_correct = 0  # TODO: fix me
    
    featureValues = dict()
    

    # Get the counts for each possible value of the class
    for index in range(n_samples):
        # Get which feature it is
        feature = x_inputs[index, feature_index]
        value   = y_outputs[index]
        # If haven't seen that feature before, make it in the dictionary
        if feature not in featureValues:
            featureValues[feature] = [0] * len(classes)
        
        # Increase the count
        featureValues[feature][classes.index(value)] += 1
        
    # Assign everything in the group the most common prediction
    for feature in featureValues.keys():
        featureValues[feature] = classes[np.argmax(featureValues[feature])]

    # Get the count of how many are correct by using this feature
    for index in range(n_samples):
        featurePrediction = featureValues[x_inputs[index, feature_index]]
        value = y_outputs[index]
        if featurePrediction == value:
            total_correct += 1

    return total_correct


def set_entropy(x_inputs, y_outputs, classes):
    """Calculate the entropy of the given input-output set.

    :param x_inputs: numpy array of shape (n_samples, n_features), containing the input data
    :param y_outputs: numpy array of shape (n_samples,), containing the output data (class labels)
    :param classes: container, unique set of class labels.
           e.g. [0,1] for two classes or [0,1,2] for 3 classes
    :return: float, entropy value of the set
    """
    n_samples, n_features = x_inputs.shape
    
    # Get the counts and divide by the total amount
    probabilities = [(np.count_nonzero(y_outputs == classVal) / n_samples) for classVal in classes]

    entropy = 0
    for prob in probabilities:
        if prob == 0:
            continue
        entropy += prob * np.log2(prob)
    entropy *= -1

    return entropy


def information_remainder(x_inputs, y_outputs, feature_index, classes):
    """Calculate the information remainder after splitting the input-output set based on the
given feature index.


    :param x_inputs: numpy array of shape (n_samples, n_features), containing the input data
    :param y_outputs: numpy array of shape (n_samples,), containing the output data (class labels)
    :param feature_index: int, index of the feature to be evaluated
    :param classes: container, unique set of class labels.
           e.g. [0,1] for two classes or [0,1,2] for 3 classes
    :return: float, information remainder value
    """

    n_samples, n_features = x_inputs.shape

    # Calculate the entropy of the overall set
    overall_entropy = set_entropy(x_inputs, y_outputs, classes)
    featureList = x_inputs[:, feature_index] # the column of feature_index (The values of the given feature)
    featureValues = list(set(featureList)) # All of the possible feature values

    # Calculate the entropy of each split set
    set_entropies = [set_entropy(x_inputs[featureList == feature], y_outputs[featureList == feature], classes) for feature in featureValues]

    # Calculate the remainder
    remainder = 0  # TODO: fix me
    for idx in range(len(featureValues)):
        feature = featureValues[idx]
        remainder += set_entropies[idx] * np.count_nonzero(featureList == feature) / n_samples

    gain = overall_entropy - remainder  # TODO: fix me

    return gain
