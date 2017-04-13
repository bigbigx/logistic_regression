# -*- coding: utf-8 -*-
"""
In this python file,I use matplotlib plotting the dividing line
"""

import matplotlib.pyplot as plt
import numpy as np
from logistic import loadDateSet, gradAscent, stocGradAscent, stocGradAscent1


def plotBestFit(weights):
    dataArr = np.array(dataMat)
    m, n = np.shape(dataArr)
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(m):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-4, 4)
    y = (-weights[0] - weights[1] * x) / weights[2]
    plt.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


dataMat, labelMat = loadDateSet()
weights = gradAscent(dataMat, labelMat)
weights_new = stocGradAscent(dataMat, labelMat)
weights_new1 = stocGradAscent1(np.array(dataMat), labelMat)
print type(weights)
print weights.getA()
print type(weights.getA())
plotBestFit(weights_new1)
