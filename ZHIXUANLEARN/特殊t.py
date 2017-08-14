# test

import numpy as np
import operator
classCount = {'2': 4, '5': 8, '9': 40}
sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
print(classCount.items())
print(sortedClassCount)