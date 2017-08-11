# -*- coding:utf-8 -*-
"""
1.运用西瓜数据集，训练处决策树，并预测结果
"""
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus
import pandas as pd


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

if __name__ == '__main__':
    # 得到西瓜数据集的数据
    df = pd.read_csv('watermellon4.2.1.csv')
    data = df.values[:10, 7:-1].tolist()
    target = df.values[:10, -1].tolist()
    # 得到测试数据
    test_data = df.values[10:, 7:-1].tolist()
    true_lables = df.values[10:, -1].tolist()
    print(data)
    print(target)
    print(test_data)
    print(true_lables)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(data, target)
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, feature_names=['density', 'suger'], class_names=['yes', 'no'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('water.pdf')
    print(caclAccuracyRate(clf, test_data, true_lables))
