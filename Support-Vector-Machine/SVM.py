#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt


def show_accuracy(a, b):
    acc = a.ravel() == b.ravel()
    print 'correct rate：%.2f%%' % (100*float(acc.sum()) / a.size)


if __name__ == "__main__":
    data = np.loadtxt(u'/Users/yunhuaxiang/Desktop/ML2/10.SVM/10.bipartition2.txt', dtype=np.float, delimiter='\t')
    x, y = np.split(data, (2, ), axis=1)
    y[y == 0] = -1
    y = y.ravel()

    #
    clfs = [svm.SVC(C=0.3, kernel='linear'),
           svm.SVC(C=10, kernel='linear'),
           svm.SVC(C=5, kernel='rbf', gamma=1),
           svm.SVC(C=5, kernel='rbf', gamma=4)]
    titles = 'linear,C=0.3', 'linear, C=10', 'rbf, gamma=1', 'rbf, gamma=4'

    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # the range of 0th column
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # the range of 1st column
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1)

    cm_light = matplotlib.colors.ListedColormap(['#77E0A0', '#FF8080'])
    cm_dark = matplotlib.colors.ListedColormap(['g', 'r'])
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10,8), facecolor='w')
    for i, clf in enumerate(clfs):
        clf.fit(x, y)

        y_hat = clf.predict(x)
        show_accuracy(y_hat, y)  #

        #
        print 'support vector number：', clf.n_support_
        print 'support vector coefficient：', clf.dual_coef_
        print 'support vector：', clf.support_
        print
        plt.subplot(2, 2, i+1)
        grid_hat = clf.predict(grid_test)
        grid_hat = grid_hat.reshape(x1.shape)
        plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light, alpha=0.8)
        plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=40, cmap=cm_dark)
        plt.scatter(x[clf.support_, 0], x[clf.support_, 1], edgecolors='k', facecolors='none', s=100, marker='o')   # support vector
        z = clf.decision_function(grid_test)
        z = z.reshape(x1.shape)
        plt.contour(x1, x2, z, colors=list('krk'), linestyles=['--', '-', '--'], linewidths=[1, 2, 1], levels=[-1, 0, 1])
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.title(titles[i])
        plt.grid()
    plt.suptitle(u'SVM classification with varying parameters', fontsize=18)
    plt.show()
