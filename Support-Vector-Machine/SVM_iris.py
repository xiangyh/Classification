#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
import matplotlib
import matplotlib.pyplot as plt


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


# 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = 'sepal length', 'sepal width', 'petal length', 'petal width'


def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print tip + 'correct rate：', float(acc.sum()) / a.size


if __name__ == "__main__":
    path = "/Users/yunhuaxiang/Desktop/ML2/4.code/4.iris.data"

    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})

    x, y = np.split(data, (4,), axis=1)

    x = x[:, :2]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)

    #
    # clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())

    #
    print clf.score(x_train, y_train)  #
    y_hat = clf.predict(x_train)
    show_accuracy(y_hat, y_train, 'train_set')
    print clf.score(x_test, y_test)
    y_hat = clf.predict(x_test)
    show_accuracy(y_hat, y_test, 'test_set')

    #
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  #

    Z = clf.decision_function(grid_test)
    print Z
    grid_hat = clf.predict(grid_test)
    print grid_hat
    grid_hat = grid_hat.reshape(x1.shape)
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.pcolormesh(x1, x2, grid_hat, cmap=plt.cm.Spectral, alpha=0.8)   
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=plt.cm.prism)      #
    plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)     #
    plt.xlabel(iris_feature[0], fontsize=13)
    plt.ylabel(iris_feature[1], fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'binary classification', fontsize=15)
    plt.grid()
    plt.show()
