# -*- coding: utf-8 -*-
classCount = {'white': 9, 'black': 10, 'red': 20, 'yellow': 7}
print(max(classCount))
print(classCount.items())
A = [1,2,3]
B = [4,7,2]
print(A + B)
newlable = 'col<=3.48'
if '<=' in newlable:
    newlable = newlable[:newlable.index('<=')]
    print(newlable)
inputTree = {"hello": 3, "get": 4}
# firstStr = inputTree.keys()[0]
print(inputTree.keys())

for i in range(2):
    print(i)