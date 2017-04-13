# -*- coding:utf-8 -*-

"""
this is a machine learning model that useing logistic regression and Sigmoid,now I build the model with python and use 
the model for predict the deadth rate of the ill horse
"""

import numpy as np
import random


def loadDateSet():
    dataMat = []
    labelMat = []
    fr = open("testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inx):
    """定义逻辑回归的模型，将训练出的模型参数与各自维度的数据值的积相加带入sigmoid函数，如果该函数的取值为>0.5该数据点被标记为1，不然被标记为0"""
    return 1.0 / (1 + np.exp(-inx))


def gradAscent(dataMatin, classLabels):
    datamatin = np.mat(dataMatin)
    classlabelsmat = np.mat(classLabels).transpose()
    # print datamatin.shape
    # print classlabelsmat.shape
    m, n = np.shape(datamatin)
    alpha = 0.001  # 步长为0.001
    maxcycles = 500  # 最大循环次数
    weights = np.ones((n, 1))  # 系数向量的初始值设为1
    for k in range(maxcycles):
        # inX = datamatin*weights  # 根据初始化的权重向量计算的测试数据集的sigmoid函数的输入值
        h = sigmoid(datamatin * weights)  # 根据初始化的inX计算的初始化的分类标签
        '''以下内容为运用梯度上升优化算法的循环过程'''
        error = (classlabelsmat - h)  # 该向量只可能由0或者1或者-1构成
        weights = weights + alpha * datamatin.transpose() * error
        # print "The shape of the error mat is {}".format(error.shape)
    return weights


def stocGradAscent(dataMatin, classLables):
    m, n = np.array(dataMatin).shape
    weights = np.ones(n)
    alpha = 0.01
    for i in range(m):
        h = sigmoid(sum(np.array(dataMatin)[i] * weights))  # 是一个类别预测值，取值为0或者-1或者1
        error = classLables[i] - h
        # 根据每一条数据来计算梯度并改进回归系数
        weights = weights + alpha * error * np.array(dataMatin)[i]
    return weights


def stocGradAscent1(dataMatin, classLabls, numIter=150):
    """
    改进后的随机梯度上升算法，alpha的值随着迭代次数的增加而减小，由于常数项的存在alpha的值不会减小到0，
    这样可以保证在多次的迭代之后新数据仍然具有一定的影响。同时在每一次迭代的过程中随机选择样本点来更新回归系数，
    这样做可以减少在回归过程中由于那些线性不可分的点造成的周期性的波动
    """
    # dataMatin = np.array(dataMatin)
    m, n = dataMatin.shape
    weights = np.ones(n)
    for j in range(numIter):
        dataindex = range(m)
        for i in range(m):
            alpha = 4.0 / (i + j + 1) + 0.01
            index = int(random.uniform(0, len(dataindex)))
            h = sigmoid(sum(dataMatin[index] * weights))
            error = classLabls[index] - h
            weights = weights + alpha * dataMatin[index] * error
            del (dataindex[index])
    return weights

