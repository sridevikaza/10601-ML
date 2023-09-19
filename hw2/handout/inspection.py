import sys
import math
import numpy as np


def getEntropy(labels):
    unique_values, unique_counts = np.unique(labels, return_counts=True)
    entropy = 0
    for i in range(len(unique_values)):
        p = unique_counts[i] / len(labels)
        entropy += -p * math.log2(p)
    return entropy


def getError(labels):
    # get majority vote
    unique_values, unique_counts = np.unique(labels, return_counts=True)
    if (unique_counts[0] > unique_counts[1]): vote = unique_values[0]
    elif (unique_counts[0] < unique_counts[1]): vote = unique_values[1]
    else: vote = max(unique_values)

    # get error rate
    predictions = np.resize(np.array(vote), labels.shape)
    return np.sum(labels != predictions) / np.size(labels)


if __name__ == '__main__':

    # find entropy and error
    labels = np.genfromtxt(sys.argv[1], skip_header=True)[:,-1]
    entropy = getEntropy(labels)
    error = getError(labels)

    # output values
    output = ["entropy: "+str(entropy), "error: "+str(error)]
    np.savetxt(sys.argv[2], output, delimiter=',',fmt='%s')