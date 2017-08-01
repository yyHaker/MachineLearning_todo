# -*- utf-8 -*-
# 学习使用DecisionTreeClassifier用于分类任务
"""
from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
print(clf.predict([[2, 2]]))
print(clf.predict([[3, 3]]))
"""
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus
iris = load_iris()
print(iris.data)
print(iris.target)
print(iris.feature_names)
print(iris.target_names)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
print(clf)
"""
# with open("iris.dot", 'w') as f:
#   f = tree.export_graphviz(clf, out_file=f)
"""
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=iris.feature_names, class_names=iris.target_names
                     , filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())
graph.write_pdf("iris.pdf")  # 写入pdf


