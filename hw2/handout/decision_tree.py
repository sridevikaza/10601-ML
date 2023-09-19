import argparse
import math
import numpy as np
# import matplotlib.pyplot as plt
import pdb

class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None
        self.zero_count = 0
        self.one_count = 0


    def train(self, train_data, attributes, max_depth, curr_depth=0):
        # save zero and one counts for printing
        self.one_count = np.sum(train_data[:,-1])
        self.zero_count = np.size(train_data[:,-1])-self.one_count

        # base case (stopping criteria)
        if curr_depth==max_depth or train_data.shape[1]==1:
            self.vote = getMajorityVote(train_data[:, -1])
            return self

        # splitting criteria
        mutual_info = getMutualInfo(train_data)
        attr_idx = np.argmax(mutual_info)
        self.attr = attributes[attr_idx]
        # pdb.set_trace()


        # get subset of left and right data from the split
        left_data = train_data[train_data[:,attr_idx]==1]
        right_data = train_data[train_data[:,attr_idx]==0]

        # create left and right child nodes
        self.left = Node()
        self.right = Node()

        # recursive call with left data subset
        if left_data.shape[0] > 0:
            self.left.train(np.delete(left_data, attr_idx, axis=1), np.delete(attributes, attr_idx), max_depth, curr_depth+1)
        else:
            self.left.vote = 1

        # recursive call with right data subset
        if right_data.shape[0] > 0:
            self.right.train(np.delete(right_data, attr_idx, axis=1), np.delete(attributes, attr_idx), max_depth, curr_depth+1)
        else:
            self.right.vote = 0

        return self  # return current node


    def test(self, test_data, attributes):
        # get the predictions for each row
        test_pred = []
        for x in test_data:
            test_pred.append(predict(self, x, attributes))

        return test_pred


def getMajorityVote(labels):
    # get unique values
    unique_values, unique_counts = np.unique(labels, return_counts=True)

    # return the max value
    if len(unique_values) == 1:
        return unique_values[0]
    elif (unique_counts[0] > unique_counts[1]):
        return unique_values[0]
    elif (unique_counts[0] < unique_counts[1]):
        return unique_values[1]
    else:
        return max(unique_values)


def predict(node, x, attributes):
    # base case (leaf node)
    if node.vote is not None:
        return node.vote
    
    # get index of the attribute in x
    attr_idx = np.where(attributes == node.attr)[0][0]

    # traverse left if attribute value is 1 and right if 0
    if x[attr_idx] == 1:
        return predict(node.left, x, attributes)
    else:
        return predict(node.right, x, attributes)


def getEntropy(labels):
    # get unique values
    unique_values, unique_counts = np.unique(labels, return_counts=True)
    entropy = 0
    
    # find probability and plug into entropy equation
    for i in range(len(unique_values)):
        p = unique_counts[i] / np.size(labels)
        entropy += -p * math.log2(p)

    return entropy


def getConditionalEntropy(labels, column):
    H = 0
    # get unique values
    unique_values, unique_counts = np.unique(column, return_counts=True)

    # iterate unique values
    for i in range(len(unique_values)):
        
        # get probability
        p = unique_counts[i] / np.size(column)

        # find probabilities and calculate entropy
        mask = column==unique_values[i]
        err = np.sum(labels[mask]) / np.sum(mask)
        if err>0 and err!=1:
            h = -err * math.log2(err) - (1-err) * math.log2(1-err)
            # pdb.set_trace()
            H += p * h

    return H


def getMutualInfo(train_data):
    # get entopy of labels
    entropy_y = getEntropy(train_data[:,-1])
    mutual_info = []

    # loop through attributes and get mutual info for each
    for attr_idx in range(train_data.shape[1]-1):
        cond_entropy = getConditionalEntropy(train_data[:,-1], train_data[:,attr_idx])
        # pdb.set_trace()
        mutual_info.append(entropy_y - cond_entropy)

    return mutual_info # return mutual info for each attribute


def getError(predictions, labels):
    # calculate ratio of correct predictions
    return np.sum(predictions != labels) / np.size(labels)


def print_tree(node, depth=0, parent_attr=None, left=None):
    # base case
    if node is None:
        return
    
    # just print stats for zero depth
    if depth==0:
        print(f"[{int(node.zero_count)} 0/{int(node.one_count)} 1]")

    # print depth, parent attribute, and counts
    else:
        print("| " * depth + f"{parent_attr} = {left}: [{int(node.zero_count)} 0/{int(node.one_count)} 1]")

    # recursively print the left and right subtrees
    print_tree(node.right, depth+1, node.attr, 0)
    print_tree(node.left, depth+1, node.attr, 1)


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, 
                        help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    parser.add_argument("metrics_out", type=str, 
                        help='path of the output .txt file to which metrics such as train and test error should be written')
    args = parser.parse_args()

    # get data
    train_data = np.genfromtxt(args.train_input, skip_header=True)
    test_data = np.genfromtxt(args.test_input, skip_header=True)
    header = np.genfromtxt(args.train_input, delimiter='\t', names=True, dtype=None, encoding='utf-8')
    attributes = np.array(header.dtype.names)

    # train decision tree
    node = Node()
    node.train(train_data, attributes, args.max_depth)
    print_tree(node)
    # pdb.set_trace()

    # test decision tree
    train_pred = node.test(train_data, attributes)
    test_pred = node.test(test_data, attributes)

    # calculate metrics
    train_error = getError(train_pred, train_data[:,-1])
    test_error = getError(test_pred, test_data[:,-1])
    metrics = ["error(train): "+str(train_error), "error(test): "+str(test_error)]

    # output values
    np.savetxt(args.train_out, train_pred, delimiter=',')
    np.savetxt(args.test_out, test_pred, delimiter=',')
    np.savetxt(args.metrics_out, metrics, delimiter=',',fmt='%s')

    # plotting (can be commented out)
    # depth = np.arange(0,len(attributes)+1)
    # test_error_plt = []
    # train_error_plt = []
    
    # for d in depth:
    #     node = Node()
    #     node.train(train_data, attributes, d)
    #     train_pred = node.test(train_data, attributes)
    #     test_pred = node.test(test_data, attributes)
    #     train_error_plt.append(getError(train_pred, train_data[:,-1]))
    #     test_error_plt.append(getError(test_pred, test_data[:,-1]))
    
    # plt.plot(depth, train_error_plt, depth, test_error_plt)
    # plt.xlabel('Depth')
    # plt.ylabel('Error')  
    # plt.title('Training vs Testing Error for Heart Dataset') 
    # plt.legend(["Training Error", "Testing Error"])
    # plt.show()