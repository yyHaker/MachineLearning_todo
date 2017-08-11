# -*- coding:utf-8 -*-
"""
定义一个MultinominalNB类，它有两个主要的方法：fit(X,y) 和 predict(X)。fit的方法其实就是训练，调用fit方法时，做的工作就是构建查找表。
predict方法就是预测，调用predict的方法时，做的工作就是求出所有的后验概率并找出最大的那个。此外，类的构造函数__init__()中，允许设定a
的值，以及设定先验概率的值。
"""
import numpy as np


class MultinominalNB(object):
    """
      Naive Bayes classfier for multinominal models
      The multinominal Naive Bayes classfier is suitable for classfication with discrete features

      Parameters
      -------------
      alpha: float, optional(default = 1.0)
                 setting alpha=0 for no smothing
                 setting 0<alpha<1 is called Lidstone smoothing
                 setting alpha = 1 is called Laplace smoothing
      fit_prior : boolean
                  whether to learn class prior probabilities or not.
                  if false , a uniform prior will be used.
      class_prior: array-like, size (n_classes,)
                  prior probabilities of the classes. If specified the priors are not
                  adjusted according to the data.(各个类别的先验概率)

      Attributes
      ------------
      fit(X, y)：
                   X and y are array-like, represent features and lables.
                   call fit() method to train Naive Bayes classfier.
       predict(X):
      """
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.classes = None
        self.conditional_prob =None

    def _caculate_feature_prob(self, feature):
        """
        计算每一维度特征的先验概率
        :param feature:某一类别的某一维度的属性值list
        :return:
        """
        values = np.unique(feature)
        total_num = float(len(feature))
        value_prob = {}
        for v in values:
            value_prob[v] = (np.sum(np.equal(feature, v)) + self.alpha) / (total_num + len(values)*self.alpha)
        return value_prob

    def fit(self, X, y):
        # TODO: check X,y
        self.classes = np.unique(y)
        # caculate class prior probabilities: p(y=ck)
        if self.class_prior == None:
            class_num = len(self.classes)
            if not self.fit_prior:
                self.class_prior = [1.0/class_num for _ in range(class_num)]   # uniform prior
            else:
                self.class_prior = []
                sample_num = float(len(y))
                for c in self.classes:
                    c_num = np.sum(np.equal(y, c))
                    self.class_prior.append((c_num + self.alpha)/(sample_num + class_num*self.alpha))
        # caculate Conditional Probability : P(xj|y=ck)
        self.conditional_prob = {}  # like {c0:{x0:{value0:0.2,value1:0.8},x1:{}},c1{...}}
        for c in self.classes:
            self.conditional_prob[c] = {}
            for i in range(len(X[0])):    # for each feature
                feature = X[np.equal(y, c)][:, i]
                self.conditional_prob[c][i] = self._caculate_feature_prob(feature)
        return self

    # given values_prob {value0:0.2, value1:0.1,value3:0.3,...} and target_value
    # return the probability of target_value
    def _get_xj_prob(self, values_prob, target_value):
        return values_prob[target_value]

    def _predict_single_sample(self, x):
        """
        predict a single sample based on (class_prior, conditional_prob)
        :param x:
        :return:
        """
        lablel = -1
        max_posterior_prob =0

        # for each category ,calculate its posterior probability: class_prior * condit
        for c_index in range(len(self.classes)):
            current_class_prior = self.class_prior[c_index]
            current_conditional_prob = 1.0
            feature_prob = self.conditional_prob[self.classes[c_index]]
            j = 0
            for feature_i in feature_prob.keys():
                current_conditional_prob *= self._get_xj_prob(feature_prob[feature_i], x[j])
                j += 1
            # compare posterior probability and update max_posterior_prob, label
            if current_class_prior * current_conditional_prob > max_posterior_prob:
                max_posterior_prob = current_class_prior * current_conditional_prob
                lablel = self.classes[c_index]
        return lablel

    # predict samples (also single sample)
    def predict(self, X):
        # TODO 1 :check and raise NoFitError
        # TODO 2 : check X
        if X.ndim == 1:
            return self._predict_single_sample(X)
        else:
            # classify each sample
            lables = []
            for i in range(X.shape[0]):
                label = self._predict_single_sample(X[i])
                lables.append(label)
            return lables

# GaussianNB differ form MultinominalNB in these two method
# _calculate_feature_prob ,_get_xj_prob
class GaussianNB(MultinominalNB):
    """
    GaussianNB inherit from MultinominalNB ,so it has self.alpha and self.fit()
    use alpha to calculate class-prior
    However, GaussianNB should calculate class_prior without alpha.
    Anyway, it make no big different
    """
    # calculate mean(mu) and standard deviation(sigma) of the given feature
    # mean deviation 平均偏差
    # standard deviation 标准差
    def _caculate_feature_prob(self, feature):
        mu = np.mean(feature)
        sigma = np.std(feature)
        return mu, sigma

    # the probability density for the Gaussian distribution
    def _prob_gaussian(self, mu, sigma, x):
        return (1.0/sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2 / (2*sigma**2))

    # given mu and sigma, return Gaussian distribution probability for target_value
    def _get_xj_prob(self, mu_sigma, target_value):
        return self._prob_gaussian(mu_sigma[0], mu_sigma[1], target_value)


# test the MultinominalNB class
if __name__ == "__main__":
    X = np.array([
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [4, 5, 5, 4, 4, 4, 5, 5, 6, 6, 6, 5, 5, 6, 6]
    ])
    X = X.T
    y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])

    nb = MultinominalNB(alpha=1.0, fit_prior=True)
    nb.fit(X, y)
    print(nb.predict(np.array([2, 4])))  # 输出-1

    nb1 = GaussianNB(alpha=0.0)
    print(nb1.fit(X, y).predict(X))



