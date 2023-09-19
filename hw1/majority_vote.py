import sys
import numpy as np


class MajorityVoteClassifier:

    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.train_pred = []
        self.test_pred = []
        self.vote = None

    def train(self):
        # get values and counts
        unique_values, unique_counts = np.unique(self.train_data[:,-1], return_counts=True)
        
        # set majority vote
        if (unique_counts[0] > unique_counts[1]): self.vote = unique_values[0]
        elif (unique_counts[0] < unique_counts[1]): self.vote = unique_values[1]
        else: self.vote = max(unique_values)

        # save training predictions
        self.train_pred = np.resize(np.array(self.vote), self.train_data[:,-1].shape)

    def predict(self):
        self.test_pred = np.resize(np.array(self.vote), self.test_data[:,-1].shape)

    def getError(self, labels, predictions):
        return np.sum(labels != predictions) / np.size(predictions)
    
if __name__ == '__main__':
    # get user inputs
    train_data = np.genfromtxt(sys.argv[1], skip_header=True)
    test_data = np.genfromtxt(sys.argv[2], skip_header=True)
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]

    # train majority classifier and predict new labels
    mvc = MajorityVoteClassifier(train_data, test_data)
    mvc.train()
    mvc.predict()

    # calculate metrics
    train_error = str(mvc.getError(mvc.train_data[:,-1], mvc.train_pred))
    test_error = str(mvc.getError(mvc.test_data[:,-1], mvc.test_pred))
    metrics = ["error(train): "+train_error, "error(test): "+test_error]

    # save values
    np.savetxt(train_out, mvc.train_pred, delimiter=',')
    np.savetxt(test_out, mvc.test_pred, delimiter=',')
    np.savetxt(metrics_out, metrics, delimiter=',',fmt='%s')