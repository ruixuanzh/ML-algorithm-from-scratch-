# Author: Ruixuan Zhang
# linkedin: https://www.linkedin.com/in/ruixuan-emily-zhang/


import numpy as np
import argparse
import math
from sklearn.tree import DecisionTreeClassifier
import csv


def parse_argument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('--train', nargs=1, required=True)
    parser.add_argument('--test', nargs=1, required=True)
    parser.add_argument('--numTrees', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


def adaboost(X, y, num_iter):
    """Given an numpy matrix X, a array y and num_iter return trees and weights

    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is in {-1, 1}^n
    """
    trees = []
    trees_weights = []
    # your code here
    n = len(y)
    w = [1. / n] * n
    for i in range(num_iter):
        h = DecisionTreeClassifier(max_depth=1)
        h = h.fit(X, y, sample_weight=w)
        trees.append(h)  # save each tree
        prediction = h.predict(X)
        compare = [1 if yhat != yy else 0 for yhat, yy in zip(prediction, y)]
        err = sum([cc * ww for cc, ww in zip(compare, w)]) / sum(w) + 10**-9
        alpha = math.log((1 - err) / err)
        trees_weights.append(alpha)  # save tree weight
        w = w * np.exp(np.asarray(compare) * alpha)
    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y
    assume Y in {-1, 1}^n
    weights: w
    """
    # your code here
    prediction_lists = [tree.predict(X) * alpha for tree, alpha in
                        zip(trees, trees_weights)]
    n = len(trees)
    prediction_result = [1 if yy > 0 else -1 for yy in
                         sum(prediction_lists) / n]
    return np.array(prediction_result)


def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row.
    """
    # your code here
    data = np.loadtxt(filename, delimiter=",")
    n = data.shape[1]
    X = data[:, :n - 1]
    Y = data[:, n - 1]
    return X, Y


def new_label(Y):
    """ Transforms a vector od 0s and 1s in -1s and 1s.
    """
    return [-1. if y == 0. else 1. for y in Y]


def old_label(Y):
    return [0. if y == -1. else 1. for y in Y]


def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))


def main():
    """
    This code is called from the command line via

    python adaboost.py --train [path to filename] --test [path to filename] --numTrees
    """
    args = parse_argument()
    train_file = args['train'][0]
    test_file = args['test'][0]
    num_trees = int(args['numTrees'][0])
    print train_file, test_file, num_trees
    # your code here
    # training set
    train_x, train_y = parse_spambase_data(train_file)
    Y = new_label(train_y)

    # train your classifiers
    mytrees, mytrees_weights = adaboost(train_x, Y, num_trees)

    # test set
    X_test, Y_test = parse_spambase_data(test_file)
    # test: convert 0 to -1
    Y_test = new_label(Y_test)

    # make prediction
    Yhat_test = adaboost_predict(X_test, mytrees, mytrees_weights)

    Yhat = adaboost_predict(train_x, mytrees, mytrees_weights)

    ## here print accuracy and write predictions to a file
    acc_test = accuracy(Y_test, Yhat_test)
    acc = accuracy(Y, Yhat)
    print "Train Accuracy %.4f" % acc
    print "Test Accuracy %.4f" % acc_test

    # write predictions back to original file
    Y_test = old_label(Y_test)
    old_yhat_test = old_label(Yhat_test)
    old_yhat_train = old_label(Yhat)
    tmp = np.concatenate((X_test, np.asarray([Y_test]).T, np.asarray([old_yhat_test]).T), axis=1)
    f_p = open('predictions.txt', 'w')
    w = csv.writer(f_p)
    w.writerows(tmp)
    f_p.close()

if __name__ == '__main__':
    main()



