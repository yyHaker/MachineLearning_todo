# -*- coding: utf-8 -*-
"""
knn的手写数字识别的算法实现
"""

from numpy import *
from os import listdir
import operator


def img2vector(filename):
    """
    样本是32*32的二值图片，将其处理成1*1024的特征向量
    :param filename:
    :return:
    """
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i + j] = int(lineStr[j])
    return returnVect


def classify0(inX, dataSet, labels, k):
    """
    分类主体程序，计算欧式距离，选择距离最小的k个，返回k个中出现频率最高的类别
    :param inX: 所要测试的向量
    :param dataSet: 训练样本集，一行对应一个样本
    :param labels: dataSet对应的标签向量
    :param k: 所选的最近邻数目
    :return:
    """
    dataSetSize = shape(dataSet)[0]   # 样本个数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet   # tile(A, (m, n ))将数组A作为元素构造m行n列的数组
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # 按照行累加
    distances = sqDistances ** 0.5
    sortedDistIndicies = argsort(distances)   # 得到每个元素的排序序号
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # get(key, x) 从字典中获取key对应的value值，没有key的话返回0
    # sorted函数，按照第二个元素即value的次序逆向排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def handwritingClassTest():
    """
    将训练集图片合成100*1024的大矩阵，同时逐一对测试集中的样本分类
    :return:
    """
    # 加载训练集到大矩阵trainingMat
    hwlabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        filenameStr = trainingFileList[i]
        fileStr = filenameStr.split('.')[0]             # 得到文件名
        classNumStr = int(fileStr.split('_')[0])  # 得到类别
        hwlabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % filenameStr)

    # 逐一读取测试图片，同时将其分类
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        testfilenameStr = testFileList[i]
        testFileStr = testfilenameStr.split('.')[0]
        testClassNumStr = int(testFileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % testfilenameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwlabels, 4)
        print("the classifier came back with : %d, the real answer is : %d" % (classifierResult, testClassNumStr))
        if classifierResult != testClassNumStr:
            errorCount += 1.0
    print("total error rate is : %f" % (errorCount/float(mTest)))

if __name__ == "__main__":
    handwritingClassTest()


