# -*- coding: utf-8 -*-
"""
In this file ,I use the Logistic_Regression to predict the death rate of the ill horses,the data is already disposed,
I replace the NUll by 0 because 0 couldn't influence the coefficient when updating,then those data that lack for classes
labels are dropped 
"""
import logistic
# import plot
import numpy as np


def classify_vector(inx, weights):
    result = logistic.sigmoid(sum(inx * weights))
    assert isinstance(result, float)
    if result >= 0.5:
        # print "The classes label is 1"
        return 1.0
    else:
        # print "The classes label is 0"
        return 0.0


def colic_test():
    fr_train = open("horseColicTraining.txt", "r")
    fr_test = open("horseColictest.txt", "r")
    training_set = []
    training_labels = []
    for line in fr_train.readlines():
        curr_line = line.strip().split("\t")
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))
    # 根据训练样本训练出模型的参数
    weights = logistic.stocGradAscent1(np.array(training_set), training_labels, 500)
    error_count = 0.0
    test_count = 0.0
    for each in fr_test.readlines():
        test_count += 1
        curr_line = each.strip().split("\t")
        vector_test = []
        for i in range(21):
            vector_test.append(float(curr_line[i]))
        if classify_vector(np.array(vector_test), weights) != float(curr_line[21]):
            error_count += 1
    error_rate = float(error_count / test_count)
    return error_rate


def multi_test():
    all_count = 10
    error_sunm = 0.0
    for i in range(all_count):
        error_sunm += colic_test()

    print "After %d test ,the averange error rate is %f" % (all_count, error_sunm / float(all_count))
    return error_sunm / float(all_count)


error = multi_test()
print error  # 综合10次测试后的预测错误率
