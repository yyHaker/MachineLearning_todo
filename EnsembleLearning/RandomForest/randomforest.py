# -*- coding: utf-8 -*-
"""
实现随机森林
"""
import pandas as pd
import math
from treeModel import *

# ETL: same procedure to training set and test set
training = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
SexCode = pd.DataFrame([1, 0], index=['female', 'male'], columns=['SexCode'])  # 将性别转换为1,0
training = training.join(SexCode, how='left', on=training.Sex)
# 删除几个不参与建模的变量，包括姓名、船票号、船舱号
training = training.drop(['Name', 'Ticket', 'Embarked', 'Cabin', 'Sex'], axis=1)
test = test.join(SexCode, how='left', on=test.Sex)
test = test.drop(['Name', 'Ticket', 'Embarked', 'Cabin', 'Sex'], axis=1)
print('ETL IS DONE!')
# print(training)
# print(test)

# model fiting
# ==========parameter adjustment=======
min_leaf = 1
min_dec_gini = 0.0001
n_trees = 20

n_fea = int(math.sqrt(len(training.columns)-1))  # 属性个数取平方根
# =====================================

# ensemble bu random forest
FOREST = {}
tmp = list(training.columns)   # 得到训练数据的属性列表 ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'SexCode']
tmp.pop(tmp.index('Survived'))  # 移除属性
feaList = pd.Series(tmp)
for t in range(n_trees):
    feasample = feaList.sample(n=n_fea, replace=False)  # select feature
    # print(feasample)
    fea = feasample.tolist()
    fea.append('Survived')
    subset = training.sample(n=len(training), replace=True)  # generate the dataSet with replacement
    subset = subset[fea]
    FOREST[t] = tree_grow(subset, 'Survived', min_leaf, min_dec_gini)  # save the tree

# model prediction
# =================
currentdata = training
output = 'submission_rf'
# ==================

prediction = {}
for r in currentdata.index:
    prediction_vote = {1: 0, 0: 0}
    row = currentdata.get(currentdata.index == r)
    for n in range(n_trees):
        tree_dict = FOREST[n]
        p = model_prediction (tree_dict, row)
        prediction_vote[p] += 1
    vote = pd.Series(prediction_vote)
    prediction[r] = list(vote.order(ascending=False).index)[0]  # the vote result
result = pd.Series(prediction, name='Survived_p')

t = training.join(result, how='left')
accuracy = round(len(t[t['Survived'] == t['Survived_p']])/len(t), 5)
print("the accuracy is:", accuracy)






