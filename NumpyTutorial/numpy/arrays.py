"""
Arrays

A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers.
The number of dimensions is the rank of the array; the shape of an array is a tuple of integers giving
the size of the array along each dimension.

We can initialize numpy arrays from nested Python lists, and access elements using square brackets:
"""

import numpy as np

a = np.array([1, 2, 3])   # Create a rank 1 array
print(type(a))              # Prints "<class 'numpy.ndarray'>"
print(a.shape)              # Prints "(3,)"
print(a[0], a[1], a[2])   # Prints "1 2 3"
a[0] = 5                      # Change an element of the array
print(a)                      # Prints "[5, 2, 3]"

b = np.array([[1, 2, 3], [4, 5, 6]])  # Create a rank 2 array
print(b.shape)                             # Prints "(2, 3)"
print(b[0, 1], b[0, 2], b[1, 0])       # Prints "2 3 4"
print(b)                                      # Prints "[[1 2 3]
                                                  #              [4 5 6]]"

a = np.zeros((2, 2))   # Create an array of all zeros
print(a)                   # Prints "[[ 0.  0.]
                               #          [ 0.  0.]]"
print(type(a))          # Prints "<class 'numpy.ndarray'> "

b = np.ones((1, 2))    # Create an array of all ones
print(b)                  # Prints "[[ 1.  1.]]"

c = np.full((2, 2), 7)  # Create a constant array
print(c)                   # Prints "[[7 7]
                              #              [7 7]]"

d = np.eye(2)           # Create a 2x2 identity matrix (单位矩阵)
print(d)                  # Prints "[[ 1.  0.]
                             #          [ 0.  1.]]"

e = np.random.random((2, 2))  # Create an array filled with random values
print(e)                                # Might print "[[ 0.91940167  0.08143941]
                                           #               [ 0.68744134  0.87236687]]"


