# -*- coding: utf-8 -*-
"""
Decision Tree Source Code for Machine Learning
algorithm:  ID3,C4.5,CART 以信息增益、增益率为准则来选择最优的划分属性
@author leyuan
"""
from math import log
import operator
import treePlotter

def createDataSet():
    """
        产生测试数据
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    计算给定数据集的信息熵(information entropy)，
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)
    labelCounts = {}
    # 统计每个类别出现的次数，保存在字典labelCounts中
    for featVec in dataSet: 
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():  # 如果当前键值不存在，则扩展字典并将当前键值加入字典
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        # 使用所有类标签的发生频率计算类别出现的概率
        prob = float(labelCounts[key])/numEntries
        # 用这个概率计算信息熵
        shannonEnt -= prob * log(prob, 2)  # 取2为底的对数
    return shannonEnt


def calcGini(dataSet):
    """
    计算给定数据集的基尼指数
    :param dataSet:
    :return:
    """
    numExample = len(dataSet)
    lableCounts = {}
    # 统计每个类别出现的次数，保存在字典lableCounts中
    for featVect in dataSet:
        currentLable = featVect[-1]
        # 如果当前键值不存在，则扩展字典将当前键值加入到字典中
        if currentLable not in lableCounts.keys():
            lableCounts[currentLable] = 0
        lableCounts[currentLable] += 1
    gini = 1.0
    for key in lableCounts:
        # 使用所有类标签的频率来计算概率
        prob = float(lableCounts[key])/numExample
        # 计算基尼指数
        gini -= prob**2
    return gini

def splitDataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集
    dataSet：待划分的数据集
    axis：   划分数据集的第axis个特征
    value：  特征的返回值（比较值）
    """
    retDataSet = []
    # 遍历数据集中的每个元素，一旦发现符合要求的值，则将其添加到新创建的列表中
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)

            # extend()和append()方法功能相似，但在处理列表时，处理结果完全不同
            # a=[1,2,3]  b=[4,5,6]
            # a.append(b) = [1,2,3,[4,5,6]]
            # a.extend(b) = [1,2,3,4,5,6]
    return retDataSet


def chooseBestFeatureToSplit(dataSet, modelType ='ID3'):
    """
    选择最好的数据集划分方式，支持ID3,C4.5,CART
    :param dataSet: 数据集
    :param modelType: 决定选择最优划分属性的方式
    :return: 最优分类的特征的index
    """
    # 计算特征数量
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    infoGainList = []
    gain_ratioList = []
    gini_index_list = []
    for i in range(numFeatures):
        # 创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        # 计算用某种属性划分的信息熵和信息增益
        newEntropy = 0.0
        instrinsicValue = 0.0
        # 基尼指数
        gini_index = 0.0
        for value in uniqueVals:
            # 计算属性的每个取值的信息熵x权重
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            # 计算固有值(instrinsic value)
            instrinsicValue -= prob * log(prob, 2)
            # 计算基尼指数
            gini_index += prob * calcGini(subDataSet)
        # 计算信息增益
        infoGain = baseEntropy - newEntropy
        infoGainList.append(infoGain)
        # 计算增益率
        if instrinsicValue == 0:
            gain_ratio = 0
        else:
            gain_ratio = infoGain/instrinsicValue
        gain_ratioList.append(gain_ratio)
        # 保存基尼指数
        gini_index_list.append(gini_index)
    # C4.5实现两个步骤:1.找出信息增益高于平均水平的属性组成集合A  2.从A中选择增益率最高的
    # 求infoGain平均值
    avgInfoGain = sum(infoGainList)/len(infoGainList)
    infoGainSublist = [gain for gain in infoGainList if gain >= avgInfoGain]


    # ID3信息增益越大能得到最优化分
    if modelType == 'ID3':
        bestInfoGain = max(infoGainList)
        bestFeature = infoGainList.index(bestInfoGain)
    # C4.5得到最优化分属性
    elif modelType == 'C4.5':
        # 选择增益率最高的
        maxGainRatio = 0.0
        for i in [infoGainList.index(infor) for infor in infoGainSublist]:
            if gain_ratioList[i] > maxGainRatio:
                maxGainRatio = gain_ratioList[i]
                bestFeature = i
    elif modelType == 'CART':
        # 选择划分后基尼指数最小的
        minGini = 1
        for i in range(len(gini_index_list)):
            if gini_index_list[i] < minGini:
                minGini = gini_index_list[i]
                bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """
    投票表决函数
    输入classList:标签集合，本例为：['yes', 'yes', 'no', 'no', 'no']
    输出：得票数最多的分类名称
    :param classList:
    :return:
    """
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 把分类结果进行排序，然后返回得票数最多的分类结果
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels, featDict):
    """
    创建树
    :param dataSet: 数据集
    :param labels: 标签列表（属性集合）
    :return:
    """
    # classList为数据集的所有类标签
    classList = [example[-1] for example in dataSet]
    # 停止条件1:所有类标签完全相同，直接返回该类标签
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
    # 停止条件2:遍历完所有特征时仍不能将数据集划分成仅包含唯一类别的分组，则返回出现次数最多的
    # 此处还存在一种情况数据集dataSet在属性集上取值相同???
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最优分类特征
    bestFeat = chooseBestFeatureToSplit(dataSet, modelType='ID3')
    bestFeatLabel = labels[bestFeat]

    # myTree存储树的所有信息
    myTree = {bestFeatLabel: {}}
    # 以下得到列表包含的所有属性值
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # 遍历当前选择特征包含的所有属性值(怎么保证该属性能取到属性的所有值？我这里在外面写了一个getFeatAllVals)
    for value in featDict[bestFeatLabel]:
        resDataSet = splitDataSet(dataSet, bestFeat, value)
        if len(resDataSet) == 0:
            myTree[bestFeatLabel][value] = majorityCnt(classList)
        else:
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = createTree(resDataSet, subLabels, featDict)
    return myTree                         


def getFeatAllVals(dataSet, lables):
    """
    获得给定数据集的指定标签的所有属性取值
    :param dataSet:
    :param lables:
    :return:
    """
    featDict = {}
    for i in range(len(lables)):
        featValues = [example[i] for example in dataSet]
        uniqueVals = set(featValues)
        featDict[lables[i]] = uniqueVals
    return featDict


def classify(inputTree, featLabels, testVec):
    """
    决策树的分类函数
    :param inputTree: 训练好的树信息
    :param featLabels: 标签列表
    :param testVec: 测试向量
    :return:
    """
    # 在2.7中，找到key所对应的第一个元素为：firstStr = myTree.keys()[0]，
    # 这在3.4中运行会报错：‘dict_keys‘ object does not support indexing，这是因为python3改变了dict.keys,
    # 返回的是dict_keys对象,支持iterable 但不支持indexable，
    # 我们可以将其明确的转化成list，则此项功能在3中应这样实现：
    firstSides = list(inputTree.keys())
    firstStr = firstSides[0]
    secondDict = inputTree[firstStr]

    # 将标签字符串转换成索引
    featIndex = featLabels.index(firstStr)

    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    # 递归遍历整棵树，比较testVec变量中的值与树节点的值，如果到达叶子节点，则返回当前节点的分类标签
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    """
    使用pickle模块存储决策树
    :param inputTree: 训练好的树信息
    :param filename:
    :return:
    """
    import pickle
    fw = open(filename, 'wb+')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    """
     导入决策树模型
    :param filename:
    :return:
    """
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)
    
if __name__ == "__main__":
    fr = open('watermellon2')
    lenses = [inst.strip().split('-') for inst in fr.readlines()]
    lensesLabels = ['color', 'root', 'stroke', 'grain', 'navel', 'touch']
    featDict = getFeatAllVals(lenses, lensesLabels)
    lensesTree = createTree(lenses, lensesLabels, featDict)
    treePlotter.createPlot(lensesTree)

    
    
