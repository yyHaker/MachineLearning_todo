# -*- coding: utf-8 -*-
"""
测试KMeans算法，问题python3.5的pickle读取数据的编码问题，在python2.7中运行无误
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
from Cluster.KMeans.kmeans import KMeans, biKMeans

if __name__ == "__main__":
    # print(sys.getdefaultencoding())
    # 加载数据
    f = open('data.pkl', 'r')
    X, y = pickle.load(f)

    # 依次画出迭代1次，迭代2次、迭代3次...图
    for max_iter in range(6):
        # 设置参数
        n_clusters = 10
        # 将初始质心初始化为X[50:60]
        initCent = X[50:60]
        # 训练模型
        clf = KMeans(n_clusters, initCent, max_iter)
        clf.fit(X)
        cents = clf.centroids
        labels = clf.labels
        sse = clf.sse
        # 画出聚类结果，每一类用一种颜色
        colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90', '#845868']
        for i in range(n_clusters):
            index = np.nonzero(labels == i)[0]
            x0 = X[index, 0]
            x1 = X[index, 1]
            y_i = y[index]
            for j in range(len(x0)):
                plt.text(x0[j], x1[j], str(int(y_i[j])), color=colors[i],
                         fontdict={'weight': 'bold', 'size': 9})
            plt.scatter(cents[i, 0], cents[i, 1], marker='x', color=colors[i], linewidths=12)
        plt.title("SSE={: .2f}".format(sse))
        plt.axis([-30, 30, -30, 30])
        plt.show()
