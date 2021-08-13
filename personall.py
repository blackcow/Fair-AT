import math

import pandas as pd
import numpy as np



if __name__ == '__main__':
    # 原始数据
    # X1 = pd.Series([1, 2, 3, 4, 5, 6])
    # Y1 = pd.Series([0.3, 0.9, 2.7, 2, 3.5, 5])

    # X1 = pd.Series([0.935,0.902,0.859,0.707])
    # Y1 = pd.Series([0.978,0.973,0.973,0.972])

    X1 = pd.Series([0.845,0.786,0.7,0.457])
    # Y1 = pd.Series([0.913,0.927,0.933,0.941])
    Y1 = pd.Series([0.959,0.962,0.966,0.968])

    # 处理数据删除Nan
    x1 = X1.dropna()
    y1 = Y1.dropna()
    n = x1.count()
    x1.index = np.arange(n)
    y1.index = np.arange(n)

    # 分部计算
    d = (x1.sort_values().index - y1.sort_values().index) ** 2
    dd = d.to_series().sum()

    p = 1 - n * dd / (n * (n ** 2 - 1))

    # s.corr()函数计算
    per = X1.corr(Y1,method="pearson")  # 皮尔森相关性系数 # 0.948136664010285
    sp = x1.corr(y1, method='spearman')
    print(per)  # 0.942857142857143 0.9428571428571428
    print(sp)  # 0.942857142857143 0.9428571428571428