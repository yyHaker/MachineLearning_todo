# -*- coding: utf-8 -*-
from numpy import *
import pylab
import random, math


def loadDataSet(fileName):
    """
    general function to parse tab -delimited floats.
    assume last column is target value.
    :param fileName:
    :return:
    """
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # map all elements to float
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat


def init_params(X, centroids, K, N, D):
    """
    初始化高斯混合分布的模型参数
    :param X: 数据矩阵
    :param centroids: K个质心的二维向量
    :param K: 高斯混合成分的个数
    :param N: 样本数据集的个数
    :param D: 样本数据的维度
    :return:  pMiu:  a K-by-D matrix.      均值
                     pPi:    a 1-by-K vector.  选择第i个混合成分的概率,k类GMM所占权重（influence factor）
                    pSigma:  a D-by-D-by-K matrix. 协方差
    """
    pMiu = centroids
    pPi = zeros((1, K))
    pSigma = zeros((D, D, K))

    # 计算距离矩阵，得到N*K的矩阵（x-pMiu）^2 = x^2+pMiu^2-2*x*Miu
    distmat = tile(sum(power(X, 2), 1), (1, K)) + \
              tile(transpose(sum(power(pMiu, 2), 1)), (N, 1)) - 2 * X * transpose(pMiu)
    # Return the minimum from each row
    labels = distmat.argmin(1)

    # 获取k类的pPi和协方差矩阵
    for k in range(K):
        boolList = (labels == k).tolist()
        indexList = [boolList.index(i) for i in boolList if i == [True]]
        Xk = X[indexList, :]
        # print cov(Xk)
        pPi[0][k] = float(size(Xk, 0))/N
        pSigma[:, :, k] = cov(transpose(Xk))

    return pMiu, pPi, pSigma


def calc_prob(pMiu, pSigma, X, K, N, D):
    """
    计算每个数据由第k类生成的概率矩阵Px
    :param pMiu: a K-by-D matrix.      均值
    :param pSigma:  a D-by-D-by-K matrix. 协方差
    :param X: 数据矩阵
    :param K: 高斯混合成分的个数
    :param N: 样本数据集的个数
    :param D: 样本数据的维度
    :return:
    """
    # Gaussian posterior probability
    Px = mat(zeros((N, K)))
    for k in range(K):
        Xshift = X - tile(pMiu[k, :], (N, 1))  # X-pMiu
        # inv_pSigma = mat(pSigma[:, :, k]).I
        inv_pSigma = linalg.pinv(mat(pSigma[:, :, k]))

        tmp = sum(array((Xshift * inv_pSigma)) * array(Xshift), 1)  # 这里应变为一列数
        tmp = mat(tmp).T
        # print linalg.det(inv_pSigma),'54545'

        Sigema = linalg.det(mat(inv_pSigma))

        if Sigema < 0:
            Sigema = 0

        coef = power((2 * math.pi), (-D / 2)) * sqrt(Sigema)   # 感觉是/???
        Px[:, k] = coef * exp(-0.5 * tmp)
    return Px


def gmm(file, K_or_centroids):
    """
    Expectation-Maximization iteration implementation of  Gaussian Mixture Model.

     PX = GMM(X, K_OR_CENTROIDS)
    [PX MODEL] = GMM(X, K_OR_CENTROIDS)
       - X: N-by-D data matrix.
       - K_OR_CENTROIDS: either K indicating the number of
            components or a K-by-D matrix indicating the
           choosing of the initial K centroids.
           K_or_centroids可以是一个整数，也可以是k个质心的二维列向量
       - PX: N-by-K matrix indicating the probability of each
            component generating each point.
       - MODEL: a structure containing the parameters for a GMM:
            MODEL.Miu: a K-by-D matrix.      均值
            MODEL.Sigma: a D-by-D-by-K matrix. 协方差
            MODEL.Pi: a 1-by-K vector.  选择第i个混合成分的概率
    :param file:
    :param K_or_centroids:
    :return:
    """
    # Gernerate Initial Centroids
    threshold = 1e-15
    dataMat = mat(loadDataSet(file))
    N, D = shape(dataMat)
    # K_or_centroids = 2
    if shape(K_or_centroids) == ():
        K = K_or_centroids
        Rn_index = range(N)
        random.shuffle(Rn_index)
        # generate K random centroids
        centroids = dataMat[Rn_index[0:K], :]
    else:
        K = size(K_or_centroids, 0)
        centroids = K_or_centroids

    # initial values
    pMiu, pPi, pSigma = init_params(dataMat, centroids, K, N, D)
    Lprev = -inf  #上一次聚类的误差

    while True:
        # Estimation Step
        Px = calc_prob(pMiu, pSigma, dataMat, K, N, D)

        # new value for pGamma(N*k), pGamma(i,k) = Xi由第k个Gaussian生成的概率
        # 或者说xi中有pGamma(i,k)是由第k个Gaussian生成的
        pGamma = mat(array(Px) * array(tile(pPi, (N, 1))))  # 分子 = pi(k) * N(xi | pMiu(k), pSigma(k))
        pGamma = pGamma / tile(sum(pGamma, 1), (1, K))  # 分母 = pi(j) * N(xi | pMiu(j), pSigma(j))对所有j求和

        # Maximization Step - through Maximize likelihood Estimation
        # print 'dtypeddddddddd:',pGamma.dtype
        Nk = sum(pGamma, 0)  # Nk(1*k) = 第k个高斯生成每个样本的概率的和，所有Nk的总和为N。

        # update pMiu and pPi
        pMiu = mat(diag((1 / Nk).tolist()[0])) * (pGamma.T) * dataMat  # update pMiu through MLE(通过令导数 = 0得到)
        pPi = Nk / N

        # update k个 pSigma
        print('kk=', K)
        for kk in range(K):
            Xshift = dataMat - tile(pMiu[kk], (N, 1))

            Xshift.T * mat(diag(pGamma[:, kk].T.tolist()[0])) * Xshift / 2

            pSigma[:, :, kk] = (Xshift.T * mat(diag(pGamma[:, kk].T.tolist()[0])) * Xshift) / Nk[kk]

        # check for convergence
        L = sum(log(Px * pPi.T))
        if L - Lprev < threshold:
            break
        Lprev = L
    return Px

