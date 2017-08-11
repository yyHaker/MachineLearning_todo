# -*- coding:utf-8 -*-
"""
使用sklearn.tree训练iris数据(150个数据，4个属性，3个类别)
实验一:使用150个数据训练出来的树，叶子节点纯度达到100%
实验二:使用100个数据训练树，剩余50个数据来测试决策树的准确率，最终准确率是96%
"""
import pandas as pd
from sklearn import tree
import pydotplus
from sklearn.externals.six import StringIO


def caclAccuracyRate(tree_clf, test_data, true_lables):
    """
    计算决策树模型预测的准确性
    :param tree_clf: 训练好的决策树
    :param test_data: 测试数据
    :param true_lables: 真实标记
    :return: 决策树模型预测的准确率
    """
    predict_lables = tree_clf.predict(test_data)
    error = 0.0
    for idx, plable in enumerate(predict_lables):
        if plable != true_lables[idx]:
            error += 1
    return 1 - error/float(len(test_data))

if __name__ == "__main__":
    # 得到数据iris.csv
    df = pd.read_csv('iris.csv')
    data = df.values[:100, :-1].tolist()
    target = df.values[:100, -1].tolist()
    # 得到测试数据
    test_data = df.values[100:, :-1].tolist()
    true_lables = df.values[100:, -1].tolist()
    print(data)
    print(target)
    # test_data
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(data, target)
    print(clf)
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                                        'petal width (cm)']
                         , class_names=['setosa', 'versicolor', 'virginica'], filled=True
                         , rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("iris01.pdf")
    print(caclAccuracyRate(clf, test_data, true_lables))
