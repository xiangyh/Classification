#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]

# 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = 'sepal length', 'sepal width', 'petal length', 'petal width'

if __name__ == "__main__":
    matplotlib.rcParams['font.sans-serif'] = [u'Times New Roman']
    matplotlib.rcParams['axes.unicode_minus'] = False

    path = "/Users/yunhuaxiang/Desktop/ML2/4.code/4.iris.data"

    #
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})

    #
    x_prime, y = np.split(data, (4,), axis=1)

    feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    plt.figure(figsize=(12, 10), facecolor='#FFFFFF')
    for i, pair in enumerate(feature_pairs):
        #
        x = x_prime[:, pair]

        #
        dtc = RandomForestClassifier(n_estimators=50, criterion='entropy')
        rf_dtc = dtc.fit(x, y)

        #
        N, M = 500, 500  #
        x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
        x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
        t1 = np.linspace(x1_min, x1_max, N)
        t2 = np.linspace(x2_min, x2_max, M)
        x1, x2 = np.meshgrid(t1, t2)  #grid
        x_test = np.stack((x1.flat, x2.flat), axis=1)  #test

        # prediction from training data
        y_hat = rf_dtc.predict(x)
        y = y.reshape(-1)
        c = np.count_nonzero(y_hat == y)
        print 'feature：  ', iris_feature[pair[0]], ' + ', iris_feature[pair[1]],
        print '\t correct predicted：', c,
        print '\t accuracy: %.2f%%' % (100 * float(c) / float(len(y)))

        # plot
        y_hat = rf_dtc.predict(x_test)  # prediction
        y_hat = y_hat.reshape(x1.shape)
        plt.subplot(2, 3, i+1)
        plt.pcolormesh(x1, x2, y_hat, cmap=plt.cm.RdYlGn, alpha=0.1)
        plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.spring)
        plt.xlabel(iris_feature[pair[0]], fontsize=14)
        plt.ylabel(iris_feature[pair[1]], fontsize=14)
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.grid()
        plt.axis('tight')
    plt.suptitle(u'Random Forest classification', fontsize=20)
    plt.show()
