# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:50:57 2016

@author: leyuan

测试西瓜数据集时，对于颜色这一属性，由于递归过程中，该颜色取值没有相关dataSet，导致左后产生的树没有相关属性分支
本例数据集中没有产生颜色取值为plain的分支
"""
import trees
import treePlotter
fr = open('watermellon2')
lenses = [inst.strip().split('-') for inst in fr.readlines()]
lensesLabels = ['color', 'root', 'stroke', 'grain', 'navel', 'touch']
featDict = trees.getFeatAllVals(lenses, lensesLabels)
lensesTree = trees.createTree(lenses, lensesLabels, featDict)
treePlotter.createPlot(lensesTree)
