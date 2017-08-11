# -*- coding:utf-8 -*-
from numpy import *
import operator
import copy
import json
from DecisionTree.yyhaker import treePlotter
import pandas as pd


def calcGini(dataSet):
    """
    计算给定数据集的基尼指数
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)
    lableCounts = {}
    # 给所有可能的分类创建字典
    for featVec in dataSet:
        currentLable = featVec[-1]
        if currentLable not in lableCounts.keys():
            lableCounts[currentLable] = 0
        lableCounts[currentLable] += 1
    Gini = 1.0
    for key in lableCounts:
        prob =float(lableCounts[key])/numEntries
        Gini -= prob * prob
    return Gini


def splitDataSet(dataSet,axis,value):
    """
    对离散变量划分数据集，取出该特征值为value的所有子集
    :param dataSet: 数据集
    :param axis: 属性下标
    :param value: 属性取值
    :return:
    """
    returnMat = []
    for data in dataSet:
        if data[axis] == value:
            returnMat.append(data[:axis]+data[axis+1:])
    return returnMat


def splitContinuousDataSet(dataSet,axis,value,direction):
    """
    对连续变量划分数据集
    :param dataSet: 数据集
    :param axis: 属性下标
    :param value: 属性值
    :param direction: 划分的方向，决定划分是小于value的样本还是大于value的样本，direction=0得到大于value的子集
    :return:
    """
    returnMat = []
    for data in dataSet:
        if direction == 0:
            if data[axis] > value:
                returnMat.append(data)
        else:
            if data[axis] <= value:
                returnMat.append(data)
    return returnMat

"""
决策树最核心的地方就是用何种方式来选择最优的划分属性？
对于这个白酒分类问题，我采用了基于基尼指数来选择最优的划分属性
从输入的训练集中，计算划分之前的基尼指数，找到当前有多少个特征，遍历每一个特征计算基尼指数，
找到能使划分后基尼指数最小的那个划分属性
这里分了两种情况：离散属性和连续属性
"""
def chooseBestFeatureToSplit(dataSet, lables):
    """
    选择最优的划分属性
    :param dataSet: 数据集
    :param lables: 属性集合
    :return: 最优划分属性的下标
    """
    numFeatures = len(dataSet[0]) - 1
    bestGini = 1000.0
    bestFeature = -1
    bestSplitDict = {}
    for i in range(numFeatures):
        # 对连续性特征进行处理，i代表第i个特征，featList是每次选取一个特征之后i这个特征的所有样本对应的数据
        featList = [example[i] for example in dataSet]
        # 对连续型值进行处理
        if type(featList[0]).__name__ == 'float' or type(featList[0]).__name__ == 'int':
            # 产生n-1个划分点
            sortFeatList = sorted(featList)
            splitList = []
            for j in range(len(sortFeatList)-1):
                splitList.append((sortFeatList[j]+sortFeatList[j+1])/2.0)
            bestSplitGini = 10000.0
            bestSplit = 100.0
            # 求用第j个候选划分点划分时，得到的信息熵，并记录最佳的划分点
            for value in splitList:
                newGini = 0.0
                subDataSet0 = splitContinuousDataSet(dataSet, i, value, 0)
                subDataSet1 = splitContinuousDataSet(dataSet, i, value, 1)
                prob0 = len(subDataSet0)/float(len(dataSet))
                newGini += prob0*calcGini(subDataSet0)
                prob1 = len(subDataSet1)/float(len(dataSet))
                newGini += prob1*calcGini(subDataSet1)
                if newGini < bestSplitGini:
                    bestSplitGini = newGini
                    bestSplit = value
            # 用字典记录当前特征的最佳划分点，记录对应的基尼指数
            bestSplitDict[lables[i]] = bestSplit
            newGini = bestSplitGini

        # 对离散型特征进行处理
        else:
            uniqueVals = set(featList)
            newGini = 0.0
            # 计算该特征下对应的信息熵，选取第i个特征的值为value的子集
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet)/float(len(dataSet))
                newGini += prob*calcGini(subDataSet)

        # 得到最优的划分属性
        if newGini < bestGini:
            bestGini = newGini
            bestFeature = i

        # 若当前节点的最优划分属性为连续属性时，则将其以之前记录的划分点为界进行二值化处理，即是否小于等于bestSplitValue
        if type(dataSet[0][bestFeature]).__name__ == 'float' or type(dataSet[0][bestFeature]).__name__ == 'int':
            bestSplitValue = round(bestSplitDict[lables[bestFeature]], 3)
            newLable = lables[bestFeature]
            if '<=' in newLable:
                newLable = newLable[:newLable.index('<=')]
                lables[bestFeature] = newLable
            lables[bestFeature] = lables[bestFeature] + '<=' + str(bestSplitValue)
        return bestFeature

def majorityCnt(classList):
    """
    特征已经划分完成，节点下的样本还没有统一取值，则需要进行投票
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def classify(inputTree, featLables, testVect):
    """
    对给定的数据集合进行分类
    :param inputTree: 训练好的决策树
    :param featLables: 属性集合
    :param testVect: 测试样本
    :return:
    """
    firstStr = list(inputTree.keys())[0]
    if u'<=' in firstStr:
        featvalue = float(firstStr.split(u'<=')[1])
        featkey = firstStr.split(u'<=')[0]
        secondDict = inputTree[firstStr]
        # 对于连续属性，我们遍历列表得到属性下标
        featIndex = 0
        for i in range(len(featLables)):
            if featkey in featLables[i]:
                featIndex = i
        if testVect[featIndex] <= featvalue:
            judge = 1
        else:
            judge = 0
        for key in secondDict.keys():
            if judge == int(key):
                if type(secondDict[key]).__name__ == 'dict':
                    classLable = classify(secondDict[key], featLables, testVect)
                else:
                    classLable = secondDict[key]
    else:  # 离散属性的情况
        secondDict = inputTree[firstStr]
        featIndex = featLables.index(firstStr)
        for key in secondDict.keys():
            if testVect[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLable = classify(secondDict[key], featLables, testVect)
                else:
                    classLable = secondDict[key]
    return classLable


def testing(myTree, data_test, lables):
    """
    后剪支
    :param myTree: 已经训练成的树
    :param data_test: 测试泛化能力的数据
    :param lables: 属性集合
    :return:
    """
    error = 0.0
    for i in range(len(data_test)):
        if classify(myTree, lables, data_test[i]) != data_test[i][-1]:
            error += 1
    return float(error)


def caclAccuracyRate(myTree, data_test, lables):
    """
    计算决策树模型预测的准确率
    :param myTree:
    :param data_test:
    :param lables:
    :return:
    """
    return 1-testing(myTree, data_test, lables)/float(len(data_test))


def testing_feat(feat, train_data, test_data, lables):
    """
    评测若选择当前最优的划分属性进行划分所产生决策树的泛化能力
    :param feat: 当前最优的划分属性
    :param train_data: 数据集
    :param test_data: 测试泛化能力的数据集
    :param lables: 属性集
    :return:
    """
    # 训练数据的类别集合
    class_list = [example[-1] for example in train_data]
    if '<=' in feat:
        featkey = feat.split('<=')[0]
    else:
        featkey = feat
    bestFeatIndex = lables.index(featkey)
    # 当前最优划分属性下标在测试数据中对应的tuple（属性取值，所属类别）
    test_data = [(example[bestFeatIndex], example[-1]) for example in test_data]
    error = 0.0

    # 判断是离散属性还是连续属性
    if '<=' in feat:
        featvalue = float(feat.split('<=')[1])   # 连续属性的划分取值
        featkey = feat.split('<=')[0]     # 连续属的名字，下标为bestFeatIndex
        # value > featvalue majority(classList0)
        subDataSet0 = splitContinuousDataSet(train_data, bestFeatIndex, featvalue, 0)
        classList0 = [example[-1] for example in subDataSet0]
        # value <= featvalue majority(classList1)
        subDataSet1 = splitContinuousDataSet(train_data,bestFeatIndex, featvalue, 1)
        classList1 = [example[-1] for example in subDataSet1]
        twoLables = [majorityCnt(classList0), majorityCnt(classList1)]
        # 计算error
        for data in test_data:
            if data[0] <= featvalue and data[1] != twoLables[1]:
                error += 1.0
            elif data[0] > featvalue and data[1] != twoLables[0]:
                error += 1.0
    else:  # 离散值
             # 当前最优划分属性的取值集合
            train_data = [example[bestFeatIndex] for example in train_data]
            all_feat = set(train_data)
            for value in all_feat:
                class_feat = [class_list[i] for i in range(len(class_list)) if train_data[i] == value]
                major = majorityCnt(class_feat)
                for data in test_data:
                    if data[0] == value and data[1] != major:
                        error += 1.0
    return error


def testingMajor(major, data_test):
    """
    评测若不选择当前最优的划分属性进行划分所产生决策树的泛化能力
    :param major:当前训练集合最多的一个类别
    :param data_test:测试泛化能力的数据集合
    :return:
    """
    error = 0.0
    for i in range(len(data_test)):
        if major != data_test[i][-1]:
            error += 1
    return float(error)


def isFeatAllTheSame(dataSet, lables):
    """
    判断数据集在属性集合上的取值是否完全相同
    :param dataSet: 数据集
    :param lables: 属性集
    :return:
    """
    for lable in lables:
        lableIndex = lables.index(lable)
        for data in dataSet:
            if data[lableIndex] != dataSet[0][lableIndex]:
                return False
    return True


def createTree(dataSet, lables, data_full, lables_full, test_data, mode="unpro"):
    """
    递归产生决策树(主程序)
    :param dataSet:数据集
    :param lables: 属性集
    :param data_full: 全部的数据
    :param lables_full: 全部的属性
    :param test_data:测试数据，用来评测泛化能力
    :param mode: 剪枝策略，不剪枝，预剪枝，后剪枝
    :return:
    """
    classList = [example[-1] for example in dataSet]
    # 数据集中的样本全部属于同一类别，将该节点标记为叶节点，并标记为该类别(注释1)
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 属性集为空或者样本数据在属性集上取值完全相同，将该节点标记为叶子节点，类别标记为样本中类别最多的一个类(注释2)
    if len(dataSet[0]) == 1 or isFeatAllTheSame(dataSet, lables):  # 样本数据dataSet在属性集上取值完全相同怎么实现？对于连续属性怎么看？
        return majorityCnt(classList)

    # 平凡情况，每次找到最佳划分特征
    lables_copy = copy.deepcopy(lables)    # 浅拷贝只得到引用，深拷贝得到具体的值
    bestFeat = chooseBestFeatureToSplit(dataSet, lables)
    bestFeatLable = lables[bestFeat]

    # 相应的剪支操作
    if mode == "unpro" or mode == "post":
        myTree = {bestFeatLable: {}}
    elif mode == "prev":
        testingfeat = testing_feat(bestFeatLable, dataSet, test_data, lables_copy)
        testingmajor = testingMajor(majorityCnt(classList), test_data)
        if testing_feat(bestFeatLable, dataSet, test_data, lables_copy) < testingMajor(majorityCnt(classList), test_data):
            myTree = {bestFeatLable: {}}
        else:
            return majorityCnt(classList)

    # 判断选择的最优的划分属性是连续属性还是离散属性
    if '<=' in bestFeatLable:  # 连续属性
        featValue = float(bestFeatLable.split("<=")[1])   # 连续属性的划分取值
        featKey = bestFeatLable.split("<=")[0]   # 连续属性的名字，下标为bestFeat

        for i in range(2):
            subDataSet = splitContinuousDataSet(dataSet, bestFeat, featValue, i)
            subClassList = [example[-1] for example in subDataSet]
            if len(subDataSet) == 0:        # 如果为空，将分支节点标记为叶节点，类别标记为父类样本中最多的类
                myTree[bestFeatLable][i] = majorityCnt(classList)
            else:
                myTree[bestFeatLable][i] = createTree(subDataSet, lables, data_full,lables_full, test_data, mode=mode )
    else:  # 离散属性
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)

        if type(dataSet[0][bestFeat]).__name__ == 'unicode' or type(dataSet[0][bestFeat]).__name__ == 'str':
            currentLable = lables_full.index(lables[bestFeat])
            featValueFull = [example[currentLable] for example in data_full]
            uniqueValsFull = set(featValueFull)
        del(lables[bestFeat])

        for value in uniqueVals:
            subLabels =lables[:]
            if type(dataSet[0][bestFeat]).__name__ == 'unicode' or type(dataSet[0][bestFeat]).__name__ == 'str':
                uniqueValsFull.remove(value)
            myTree[bestFeatLable][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, data_full,lables_full,splitDataSet(test_data, bestFeat, value),mode=mode)

        if type(dataSet[0][bestFeat]).__name__ == 'unicode' or type(dataSet[0][bestFeat]).__name__ == 'str':
            for value in uniqueValsFull:
                myTree[bestFeatLable][value] = majorityCnt(classList)

    # 后剪支
    if mode == 'post':
        if testing(myTree, test_data, lables_copy) > testingMajor(majorityCnt(classList), test_data):
            return majorityCnt(classList)
    return myTree


if __name__ == "__main__":
    # 得到数据 whitequality_white.csv
    """
     df = pd.read_csv('winequality-white.csv')  # 4898条数据
    data = df.values[:2000, :].tolist()
    test_data = df.values[2000:, :].tolist()
    data_full = data[:]
    lables = df.columns.values[:-1].tolist()
    lables_full = lables[:]
    """
    # 得到数据 adult.csv
    df = pd.read_csv('iris.csv')
    data = df.values[:100, :].tolist()
    test_data = df.values[100:, :].tolist()
    data_full = data[:]
    lables = df.columns.values[:-1].tolist()
    lables_full = lables[:]

    print(data)
    print(test_data)
    mode = "prev"
    # mode = "unpro"
    myTree = createTree(data, lables, data_full, lables_full, test_data,mode=mode)
    print(myTree)
    print("accuracyRate:", caclAccuracyRate(myTree, test_data, lables_full))
    treePlotter.createPlot(myTree)
    """
    # 数据测试(watermellon4.2.1.csv)
    df = pd.read_csv('watermellon4.2.1.csv')
    data = df.values[:11, 1:].tolist()
    test_data = df.values[11:, 1:].tolist()
    data_full = data[:]
    lables = df.columns.values[1:-1].tolist()
    lables_full = lables[:]
    """
    # 为了代码的简洁，将预剪枝，后剪枝和未剪枝三种模式用一个参数mode传入建树的过程
    # post代表后剪枝，prev代表预剪枝，unpro代表不剪枝
    """
    mode = "unpro"
    # mode = "prev"
    # mode = "post"
    # mode = "prev"
    myTree = createTree(data, lables, data_full, lables_full, test_data, mode=mode)
    # myTree = postPruningTree(myTree,train_data,test_data,labels_full)
    print(myTree)
    print(json.dumps(myTree, ensure_ascii=False, indent=4))
    print("accuracyRate:", caclAccuracyRate(myTree, test_data, lables_full))
    treePlotter.createPlot(myTree)
    """








