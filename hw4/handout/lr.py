import numpy as np
import argparse
import matplotlib.pyplot as plt
import pdb


def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)

def log_likelihood(theta, X, y):
    N = X.shape[0]
    sum = 0
    for i in range(N):
        p = pow(sigmoid(theta.T @ X[i,:]), y[i]) * pow(1-sigmoid(theta.T @ X[i,:]),(1-y[i]))
        sum += np.log(p)
        # pdb.set_trace()
    return -1/N*sum


def train(X : np.ndarray, y : np.ndarray, num_epoch : int, learning_rate : float, X_val, y_val ) -> np.ndarray:
    # initialize weights (and fold in intercept)
    N,D = X.shape
    theta = np.zeros(D+1)
    theta1 = np.zeros(D+1)
    theta2 = np.zeros(D+1)
    theta3 = np.zeros(D+1)
    X = np.hstack([np.ones((X.shape[0],1)), X])
    # X_val = np.hstack([np.ones((X_val.shape[0],1)), X_val])
    # val_nll = []
    # train_nll = []
    # train_nll_1 = []
    # train_nll_2 = []
    # train_nll_3 = []

    # SGD
    for epoch in range(num_epoch):
        for i in range(N):
            dJ = (sigmoid(theta.T @ X[i,:]) - y[i]) * X[i,:]
            theta -= dJ * learning_rate
        #     theta1 -= dJ * 0.1
        #     theta2 -= dJ * 0.01
        #     theta3 -= dJ * 0.001
        # val_nll.append(log_likelihood(theta,X_val,y_val))
        # train_nll.append(log_likelihood(theta,X,y))
        # train_nll_1.append(log_likelihood(theta1,X,y))
        # train_nll_2.append(log_likelihood(theta2,X,y))
        # train_nll_3.append(log_likelihood(theta3,X,y))

    # epochs = np.arange(0,num_epoch)
    # plt.plot(epochs, train_nll, epochs, val_nll)
    # plt.xlabel('Number of Epochs')
    # plt.ylabel('Average Negative Log Likelihood')  
    # plt.title('Average Negative Log Likelihood Over 1000 Epochs') 
    # plt.legend(["Training Dataset", "Validation Dataset"])
    # plt.show()

    # plt.plot(epochs, train_nll_1, epochs, train_nll_2, epochs, train_nll_3)
    # plt.xlabel('Number of Epochs')
    # plt.ylabel('Average Negative Log Likelihood')  
    # plt.title('Average Negative Log Likelihood Over 1000 Epochs') 
    # plt.legend(["Learning Rate = 0.1", "Learning Rate = 0.01", "Learning Rate = 0.001"])
    # plt.show()

    return theta


def predict(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.hstack([np.ones((X.shape[0],1)), X])
    return (sigmoid(X @ theta) > 0.5).astype(int)


def compute_error(y_pred: np.ndarray, y: np.ndarray) -> float:
    return np.sum(y_pred != y) / np.size(y)


def lr(train_data, test_data, val_data, num_epoch, learning_rate, train_out, test_out, metrics_out):

    # parse datasets
    train_X = train_data[:,1:]
    train_y = train_data[:,0]
    test_X = test_data[:,1:]
    test_y = test_data[:,0]
    val_X = val_data[:,1:]
    val_y = val_data[:,0]

    # train
    theta = train(train_X, train_y, num_epoch, learning_rate, val_X, val_y)
    
    # predict
    train_pred = predict(theta, train_X)
    test_pred = predict(theta, test_X)

    # metrics
    train_err = compute_error(train_pred, train_y)
    test_err = compute_error(test_pred, test_y)
    metrics = ["error(train): "+str(train_err), "error(test): "+str(test_err)]

    # output values
    np.savetxt(train_out, train_pred, delimiter='\n', fmt='%d')
    np.savetxt(test_out, test_pred, delimiter='\n', fmt='%d')
    np.savetxt(metrics_out, metrics, delimiter=',',fmt='%s')


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

    train_data = np.genfromtxt(args.train_input)
    test_data = np.genfromtxt(args.test_input)
    val_data = np.genfromtxt(args.validation_input)
    lr(train_data, test_data, val_data, args.num_epoch, args.learning_rate, args.train_out, args.test_out, args.metrics_out)