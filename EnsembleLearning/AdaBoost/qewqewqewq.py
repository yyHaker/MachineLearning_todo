# test
from numpy import *

A = mat([1,1,1,1,1])
B = mat([1,1,1,1,1])
D = array([1,1,1,1,1,1])
C = mat(ones((5,1))).T
C[D == A] = 0
print(C)
