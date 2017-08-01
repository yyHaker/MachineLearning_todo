# coding: utf-8
from numpy import *
import pandas as pd
import codecs
import operator
import copy
import json
import treePlotter


def calcGini(dataSet):
    """
    计算给定数据集的基尼指数
    :param dataSet: 数据集 list
    :return:
    """
    numEntries = len(dataSet)
    labelCounts = {}
    # 给所有可能的分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    Gini = 1.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        Gini -= prob * prob
    return Gini


def splitDataSet(dataSet,axis,value):
    """
        对离散变量划分数据集，取出该特征值为value的所有样本
        :param dataSet: 数据集list
        :param axis: 属性下标
        :param value: 属性取值
        :return:retDataSet
        """
    returnMat = []
    for data in dataSet:
        if data[axis] == value:
            returnMat.append(data[:axis]+data[axis+1:])
    return returnMat

"""
注意到连续属性和离散属性不同，对离散属性划分数据集时会删除对应属性的数据，若当前节点划分属性为连续属性，
该属性还可作为其后代节点的划分属性，因此对连续变量划分数据集时并没有删除对应属性的数据
"""
def splitContinuousDataSet(dataSet, axis, value, direction):
    """
     对连续变量划分数据集
     :param dataSet: 数据集
     :param axis: 属性下标
     :param value: 属性值
     :param direction: 划分的方向，决定划分是小于value的数据样本还是大于value 的数据样本
                             direction=0得到大于value的数据集
     :return: retDataSet
     """
    retDataSet = []
    for featVec in dataSet:
        if direction == 0:
            if featVec[axis] > value:
                retDataSet.append(featVec)
        else:
            if featVec[axis] <= value:
                retDataSet.append(featVec)
    return retDataSet

'''
决策树算法中比较核心的地方，究竟是用何种方式来决定最佳划分？
使用信息增益作为划分标准的决策树称为ID3
使用信息增益比作为划分标准的决策树称为C4.5，甚至综合信息增益和信息增益比
本题为CART基于基尼指数
从输入的训练样本集中，计算划分之前的熵，找到当前有多少个特征，遍历每一个特征计算信息增益，找到这些特征中能带来信息增益最大的那一个特征。
这里用分了两种情况，离散属性和连续属性
1、离散属性，在遍历特征时，遍历训练样本中该特征所出现过的所有离散值，假设有n种取值，那么对这n种我们分别计算每一种的熵，最后将这些熵加起来
就是划分之后的信息熵
2、连续属性，对于连续值就稍微麻烦一点，首先需要确定划分点，用二分的方法确定（连续值取值数-1）个切分点。遍历每种切分情况，对于每种切分，
计算新的信息熵，从而计算增益，找到最大的增益。
假设从所有离散和连续属性中已经找到了能带来最大增益的属性划分，这个时候是离散属性很好办，直接用原有训练集中的属性值作为划分的值就行，但是连续
属性我们只是得到了一个切分点，这是不够的，我们还需要对数据进行二值处理。
'''


def chooseBestFeatureToSplit(dataSet, labels):
    """
    选择最优的划分属性
    :param dataSet: 数据集list
    :param labels: 属性集合
    :return: 最优划分属性的下标
    """
    numFeatures = len(dataSet[0]) - 1
    bestGini = 10000.0
    bestFeature = -1
    bestSplitDict = {}
    for i in range(numFeatures):
        # 对连续型特征进行处理 ,i代表第i个特征,featList是每次选取一个特征之后这个特征的所有样本对应的数据
        featList = [example[i] for example in dataSet]
        # 对连续型值处理
        if type(featList[0]).__name__ == 'float' or type(featList[0]).__name__ == 'int':
            # 产生n-1个候选划分点
            sortfeatList = sorted(featList)
            splitList = []
            for j in range(len(sortfeatList) - 1):
                splitList.append((sortfeatList[j] + sortfeatList[j + 1]) / 2.0)
            bestSplitGini = 10000
            # 求用第j个候选划分点划分时，得到的信息熵，并记录最佳划分点
            for value in splitList:
                newGini = 0.0
                subDataSet0 = splitContinuousDataSet(dataSet, i, value, 0)
                subDataSet1 = splitContinuousDataSet(dataSet, i, value, 1)
                prob0 = len(subDataSet0) / float(len(dataSet))
                newGini += prob0 * calcGini(subDataSet0)
                prob1 = len(subDataSet1) / float(len(dataSet))
                newGini += prob1 * calcGini(subDataSet1)
                if newGini < bestSplitGini:
                    bestSplitGini = newGini
                    bestSplit = value
            # 用字典记录当前特征的最佳划分点，记录对应的基尼指数
            bestSplitDict[labels[i]] = bestSplit
            newGini = bestSplitGini

        # 对离散型特征进行处理
        else:
            uniqueVals = set(featList)
            newGini = 0.0
            # 计算该特征下划分的信息熵,选取第i个特征的值为value的子集
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newGini += prob * calcGini(subDataSet)

        # 得到最优的划分属性
        if newGini < bestGini:
            bestGini = newGini
            bestFeature = i

    # 若当前节点的最佳划分特征为连续特征，则将其以之前记录的划分点为界进行二值化处理即是否小于等于bestSplitValue
    # 问题：为什么要进行二值化处理，怎么保证如果选择的当前划分属性为连续属性，该属性还可以作为后代的划分属性
    # 思路：能不能在选择的划分属性为连续属性时除了返回属性下标外，还返回划分数值，后面再递归求解构造树
    if type(dataSet[0][bestFeature]).__name__ == 'float' or type(dataSet[0][bestFeature]).__name__ == 'int':
        bestSplitValue = round(bestSplitDict[labels[bestFeature]], 3)
        newlable = lables[bestFeature]
        if '<=' in newlable:
            newlable = newlable[:newlable.index('<=')]
            lables[bestFeature] = newlable
        labels[bestFeature] = labels[bestFeature] + '<=' + str(bestSplitValue)
    return bestFeature


def majorityCnt(classList):
    """
    特征已经划分完成，节点下的样本还没有统一取值，则需要进行投票
    :param classList:
    :return:
    """
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 由于在Tree中，连续值特征的名称以及改为了feature <= value的形式
# 因此对于这类特征，需要利用正则表达式进行分割，获得特征名以及分割阈值
def classify(inputTree, featLabels, testVec):
    """
     对给定的数据集合进行分类
    :param inputTree:训练好i的决策树
    :param featLabels:属性集合
    :param testVec: 测试样本
    :return:
    """
    firstStr = list(inputTree.keys())[0]
    classLabel = ""
    if u'<=' in firstStr:
        featvalue = float(firstStr.split(u"<=")[1])
        featkey = firstStr.split(u"<=")[0]
        secondDict = inputTree[firstStr]
        # 对于连续属性，我们遍历列表得到属性下标
        featIndex = 0
        for i in range(len(featLabels)):
            if featkey in featLabels[i]:
                featIndex = i
        if testVec[featIndex] <= featvalue:
            judge = 1
        else:
            judge = 0
        for key in secondDict.keys():
            if judge == int(key):
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel = secondDict[key]
    else:    # 离散属性的情况
        secondDict = inputTree[firstStr]
        featIndex = featLabels.index(firstStr)
        for key in secondDict.keys():
            if testVec[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel = secondDict[key]
    return classLabel


def testing(myTree, data_test, labels):
    """
    后剪枝
    :param myTree: 已经训练成的树
    :param data_test: 测试泛化能力的数据
    :param labels: 属性集
    :return:
    """
    error = 0.0
    for i in range(len(data_test)):
        if classify(myTree, labels, data_test[i]) != data_test[i][-1]:
            error += 1
    return float(error)


def caclAccuracyRate(mtTree, data_test, lables):
    """
    计算决策树模型预测的准确率
    :param mtTree:
    :param data_test:
    :param lables:
    :return:
    """
    return 1 - testing(myTree, data_test, lables)/float(len(data_test))


def testing_feat(feat, train_data, test_data, labels):
    """
    评测若选择当前最优的划分属性进行划分所产生决策树的泛化能力
    :param feat: 当前最优的划分属性
    :param train_data: 数据集
    :param test_data: 测试泛化能力的数据集
    :param labels: 属性集
    :return:
    """
    # 训练数据的类别集合
    class_list = [example[-1] for example in train_data]
    if "<=" in feat:
        featName = feat.split("<=")[0]
    else:
        featName = feat

    bestFeatIndex = lables.index(featName)
    # 当前最优化分属性下标在测试数据中对应的turple(属性取值，所属类别)
    test_data = [(example[bestFeatIndex], example[-1]) for example in test_data]
    error = 0.0

    # 判断是离散属性还是连续属性
    if "<=" in feat:  # 连续属性
        featvalue = float(feat.split("<=")[1])  # 连续属性的划分取值
        featkey = feat.split("<=")[0]  # 连续属性的名字,下标为 bestFeatIndex
        # value > featvalue  majority(classList0)
        subDataSet0 = splitContinuousDataSet(train_data, bestFeatIndex, featvalue, 0)
        classList0 =[example[-1] for example in subDataSet0]
        # value <= featvalue majority(classList1)
        subDataSet1 = splitContinuousDataSet(train_data, bestFeatIndex, featvalue, 1)
        classList1 = [example[-1] for example in subDataSet1]
        twoLables = [majorityCnt(classList0), majorityCnt(classList1)]
        # 计算error
        for data in test_data:
            if data[0] <= featvalue and data[1] != twoLables[1]:
                error += 1.0
            elif data[0] > featvalue and data[1] != twoLables[0]:
                error +=1.0
    else:  # 离散属性
        # 当前最优划分属性的取值集合
        train_data = [example[bestFeatIndex] for example in train_data]
        all_feat = set(train_data)
        for value in all_feat:
            class_feat = [class_list[i] for i in range(len(class_list)) if train_data[i] == value]
            major = majorityCnt(class_feat)
            for data in test_data:
                if data[0] == value and data[1] != major:
                    error += 1.0
    # print 'myTree %d' % error
    return error


def testingMajor(major, data_test):
    """
    评测若不选择当前最优的划分属性进行划分所产生决策树的泛化能力
    :param major: 当前训练集合最多的类别
    :param data_test: 测试泛化能力的数据集
    :return:
    """
    error = 0.0
    for i in range(len(data_test)):
        if major != data_test[i][-1]:
            error += 1
    # print 'major %d' % error
    return float(error)
'''
主程序，递归产生决策树。
params:
dataSet:用于构建树的数据集,最开始就是data_full，然后随着划分的进行越来越小，第一次划分之前是17个瓜的数据在根节点，然后选择第一个bestFeat是纹理
纹理的取值有清晰、模糊、稍糊三种，将瓜分成了清晰（9个），稍糊（5个），模糊（3个）,这个时候应该将划分的类别减少1以便于下次划分
labels：还剩下的用于划分的类别
data_full：全部的数据
label_full:全部的类别
既然是递归的构造树，当然就需要终止条件，终止条件有三个：
1、当前节点包含的样本全部属于同一类别；-----------------注释1就是这种情形
2、当前属性集为空，即所有可以用来划分的属性全部用完了，这个时候当前节点还存在不同的类别没有分开，这个时候我们需要将当前节点作为叶子节点，
同时根据此时剩下的样本中的多数类（无论几类取数量最多的类）-------------------------注释2就是这种情形
3、当前节点所包含的样本集合为空。比如在某个节点，我们还有10个西瓜，用大小作为特征来划分，分为大中小三类，10个西瓜8大2小，因为训练集生成
树的时候不包含大小为中的样本，那么划分出来的决策树在碰到大小为中的西瓜（视为未登录的样本）就会将父节点的8大2小作为先验同时将该中西瓜的
大小属性视作大来处理。
'''
def createTree(dataSet, labels, data_full, labels_full, test_data, mode="unpro"):
    """
    递归的产生决策树
    :param dataSet: 数据集
    :param labels: 属性集
    :param data_full: 全部的数据
    :param labels_full: 全部的属性
    :param test_data: 测试数据，用来评测泛化能力
    :param mode:剪枝策略，不剪枝，预剪枝，后剪枝
    :return:
    """
    classList=[example[-1] for example in dataSet]
    # 数据集中的样本全部属于同一类别，将该节点标记为叶节点，并标记为该类别(注释1)
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 属性集为空或者样本数据在属性集上完全相同，将该节点标记为叶子结点，类别标记为样本中类别最多的一个类(注释2)
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 平凡情况，每次找到最佳划分的特征
    labels_copy = copy.deepcopy(labels)    # 浅拷贝只得到引用,深拷贝得到具体的值
    bestFeat=chooseBestFeatureToSplit(dataSet, labels)
    bestFeatLabel = labels[bestFeat]

    # 相应的剪枝操作
    if mode == "unpro" or mode == "post":
        myTree = {bestFeatLabel: {}}
    elif mode == "prev":
        testingfeat = testing_feat(bestFeatLabel, dataSet, test_data, labels_copy)
        testingmajor = testingMajor(majorityCnt(classList), test_data)
        if testing_feat(bestFeatLabel, dataSet, test_data, labels_copy) < testingMajor(majorityCnt(classList), test_data):
            myTree = {bestFeatLabel: {}}
        else:
            return majorityCnt(classList)

    # 判断选择的最优的划分属性是连续属性还是离散属性
    if '<=' in bestFeatLabel:   # 连续属性
        featvalue = float(bestFeatLabel.split("<=")[1])  # 连续属性的划分取值
        featkey = bestFeatLabel.split("<=")[0]             # 连续属性的名字,下标为 bestFeat

        for i in range(2):
            subDataSet = splitContinuousDataSet(dataSet, bestFeat, featvalue, i)
            subClassList = [example[-1] for example in subDataSet]
            if len(subDataSet) == 0 or len(set(subClassList)) == 1:
                myTree[bestFeatLabel][i] = majorityCnt(subClassList)
            else:
                myTree[bestFeatLabel][i] = createTree(subDataSet, lables, data_full, lables_full, test_data, mode=mode)

    else:  # 离散属性
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)

        '''
        刚开始很奇怪为什么要加一个uniqueValFull，后来思考下觉得应该是在某次划分，比如在根节点划分纹理的时候，将数据分成了清晰、模糊、稍糊三块
        ，假设之后在模糊这一子数据集中，下一划分属性是触感，而这个数据集中只有软粘属性的西瓜，这样建立的决策树在当前节点划分时就只有软粘这一属性了，
        事实上训练样本中还有硬滑这一属性，这样就造成了树的缺失，因此用到uniqueValFull之后就能将训练样本中有的属性值都囊括。
        如果在某个分支每找到一个属性，就在其中去掉一个，最后如果还有剩余的根据父节点投票决定。
        但是即便这样，如果训练集中没有出现触感属性值为“一般”的西瓜，但是分类时候遇到这样的测试样本，那么应该用父节点的多数类作为预测结果输出。
        '''
        if type(dataSet[0][bestFeat]).__name__ == 'unicode' or type(dataSet[0][bestFeat]).__name__ == 'str':
            currentlabel = labels_full.index(labels[bestFeat])
            featValuesFull = [example[currentlabel] for example in data_full]
            uniqueValsFull = set(featValuesFull)

        del(labels[bestFeat])

        '''
        针对bestFeat的每个取值，划分出一个子树。对于纹理，树应该是{"纹理"：{？}}，显然？处是纹理的不同取值，有清晰模糊和稍糊三种，对于每一种情况，
        都去建立一个自己的树，大概长这样{"纹理"：{"模糊"：{0},"稍糊"：{1},"清晰":{2}}}，对于0\1\2这三棵树，每次建树的训练样本都是值为value特征数减少1
        的子集。
        '''
        for value in uniqueVals:
            subLabels = labels[:]
            # print(type(dataSet[0][bestFeat]+" "+dataSet[0][bestFeat]).__name__)
            if type(dataSet[0][bestFeat]).__name__ == 'unicode' or type(dataSet[0][bestFeat]).__name__ == 'str':
                uniqueValsFull.remove(value)
            myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, data_full, labels_full, splitDataSet(test_data, bestFeat, value), mode=mode)
        if type(dataSet[0][bestFeat]).__name__ == 'unicode' or type(dataSet[0][bestFeat]).__name__ == 'str':
            for value in uniqueValsFull:
                myTree[bestFeatLabel][value] = majorityCnt(classList)
    # 后剪枝
    if mode == "post":
        if testing(myTree, test_data, labels_copy) > testingMajor(majorityCnt(classList), test_data):
            return majorityCnt(classList)
    return myTree


# 读入csv文件数据
def load_data(file_name):
    file = codecs.open(file_name, "r", 'utf-8')
    filedata = [line.strip('\n').split(',') for line in file]
    filedata = [[float(i) if '.' in i else i for i in row] for row in filedata]  # change decimal from string to float
    train_data = [row[1:] for row in filedata[1:12]]
    test_data = [row[1:] for row in filedata[11:]]
    labels = []
    for label in filedata[0][1:-1]:
        labels.append(unicode(label))
    return train_data,test_data,labels


if __name__ == "__main__":
    """
    train_data,test_data,labels = load_data("data/西瓜数据集2.0.csv")
    data_full = train_data[:]
    labels_full = labels[:]
    """

    """
    # 数据测试
    df = pd.read_csv('watermellon4.2.1.csv')
    data = df.values[:11, 1:].tolist()
    test_data = df.values[11:, 1:].tolist()
    data_full = data[:]
    lables = df.columns.values[1:-1].tolist()
    lables_full = lables[:]

     # 得到数据
    df = pd.read_csv('winequality-white.csv')  # 4898条数据
    data = df.values[:2000, :].tolist()
    test_data = df.values[2000:, :].tolist()
    data_full = data[:]
    lables = df.columns.values[:-1].tolist()
    lables_full = lables[:]

    # 为了代码的简洁，将预剪枝，后剪枝和未剪枝三种模式用一个参数mode传入建树的过程
    # post代表后剪枝，prev代表预剪枝，unpro代表不剪枝
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

    # 得到数据 adult.csv
    df = pd.read_csv('adult.csv')
    data = df.values[:200, :].tolist()
    test_data = df.values[200:, :].tolist()
    data_full = data[:]
    lables = df.columns.values[:-1].tolist()
    lables_full = lables[:]

    print(data)
    print(test_data)
    mode = "prev"
    # mode = "unpro"
    myTree = createTree(data, lables, data_full, lables_full, test_data, mode=mode)
    print(myTree)
    print("accuracyRate:", caclAccuracyRate(myTree, test_data, lables_full))
    treePlotter.createPlot(myTree)
