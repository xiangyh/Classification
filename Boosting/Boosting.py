# /usr/bin/python
# -*- encoding:utf-8 -*-


import xgboost as xgb
import numpy as np

# define f: theta * x
def log_reg(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label()
    h = p * (1.0-p)
    return g, h


def error_rate(y_hat, y):
    return 'error', float(sum(y.get_label() != (y_hat > 0.0))) / len(y_hat)


if __name__ == "__main__":
    #
    path = "/Users/yunhuaxiang/Desktop/ML2/8.data/"
    data_train = xgb.DMatrix(path + '8.agaricus_train.txt')
    data_test = xgb.DMatrix(path + '8.agaricus_test.txt')

    #
    # param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'} # logitraw
    param = {'max_depth': 2, 'eta': 1, 'silent': 1}
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    n_round = 2
    # bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist, obj=log_reg, feval=error_rate)

    #
    y_hat = bst.predict(data_test)
    y = data_test.get_label()
    print y_hat
    print y
    error = sum(y != (y_hat > 0))
    error_rate = float(error) / len(y_hat)
    print 'sample size：\t', len(y_hat)
    print 'error number：\t%4d' % error
    print 'error rate：\t%.2f%%' % (100*error_rate)

