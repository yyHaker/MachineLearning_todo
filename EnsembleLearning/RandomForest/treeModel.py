# -*- coding:utf-8 -*-
"""
使用Gini指数来作为决策树选择最优划分属性的标准
"""
import pandas as pd
import numpy as np
import math


def gini_cal(class_lable):
    """
    计算数据集的基尼指数，这里只针对0/1的二分类任务
    :param class_lable: 类别标签0/1
    :return:
    """
    p_1 = sum(class_lable)/len(class_lable)
    p_0 = 1 - p_1
    gini = 1 - (pow(p_0, 2) + pow(p_1, 2))
    return gini


def dataSplit(dataFrame, split_fea, split_val):
    """
    按照给定的spli_val对数据集进行分割
    :param dataFrame: 数据集
    :param split_fea: 数据下标
    :param split_val: 分割值
    :return:
    """
    left_node = dataFrame[dataFrame[split_fea] <= split_val]
    right_node = dataFrame[dataFrame[split_fea] > split_val]
    return left_node, right_node


def best_split_col(dataFrame, target_name):
    """
    基于数据集dataFrame选择最优的划分属性
    :param dataFrame:
    :param target_name:
    :return: 最优划分属性的下标，最优划分属性的值， 基尼指数减少量
    """
    best_fea = ''
    best_split_point = 0
    col_list = list(dataFrame.columns)
    col_list.remove(target_name)
    gini_0 = gini_cal(dataFrame[target_name])
    n = len(dataFrame)
    gini_dec = -99999999
    for col in col_list:
        node = dataFrame[[col, target_name]]
        unique = node.groupby(col).count().index
        for split_point in unique:
            left_node, right_node = dataSplit(node, col, split_point)
            if len(left_node) > 0 and len(right_node) > 0:
                gini_col = gini_cal(left_node[target_name])*(len(left_node)/n)+gini_cal(right_node[target_name])*(len(right_node)/n)
                if gini_0 - gini_col > gini_dec:
                    gini_dec = gini_0 - gini_col  # decrease of impurity
                    best_fea = col
                    best_split_point = split_point
    return best_fea, best_split_point, gini_dec


def leaf(class_label):
    """
    返回类别标签中最多的那一类
    :param class_label:
    :return:
    """
    tmp = {}
    for i in class_label:
        if i in tmp:
            tmp[i] += 1
        else:
            tmp[i] = 1
    s = pd.Series(tmp)
    s.sort(ascending=False)
    return s.index[0]


def tree_grow(dataFrame, target, min_leaf, min_dec_gini):
    """

    :param dataFrame: 数据集
    :param target:
    :param min_leaf: 叶子最小个数
    :param min_dec_gini: 减少的最少基尼指数
    :return:
    """
    tree = {}  # renew a tree
    is_not_leaf = (len(dataFrame) > min_leaf)
    if is_not_leaf:
        fea, sp, gd = best_split_col(dataFrame, target)
        if gd > min_dec_gini:
            tree['fea'] = fea  # 最优划分属性的下标
            tree['val'] = sp      # 最优划分属性的值
            l, r = dataSplit(dataFrame, fea, sp)
            l.drop(fea, axis=1)
            r.drop(fea, axis=1)
            tree['left'] = tree_grow(l, target, min_leaf, min_dec_gini)
            tree['right'] = tree_grow(r, target, min_leaf, min_dec_gini)
        else:
            return leaf(dataFrame[target])
    else:
        return leaf(dataFrame[target])
    return tree


def model_prediction(model, row):
    """
    预测数据的类别
    :param model:
    :param row: row is a df
    :return:
    """
    fea = model['fea']
    val = model['val']
    left = model['left']
    right = model['right']
    if row[fea].tolist()[0] <= val:
        branch = left
    else:
        branch = right
    if 'dict' in str(type(branch)):
        prediction = model_prediction(branch, row)
    else:
        prediction = branch
    return prediction
