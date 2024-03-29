"""
Array indexing
Numpy offers several ways to index into arrays.
"""
import numpy as np

"""
Slicing: Similar to Python lists, numpy arrays can be sliced. Since arrays may
be multidimensional, you must specify a slice for each dimension of the array:
"""
# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]
print(b)
# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(a[0, 1])   # Prints "2"
b[0, 0] = 77    # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])   # Prints "77"

"""
You can also mix integer indexing with slice indexing. However, doing so will yield an array
 of lower rank than the original array. Note that this is quite different from the way
 that MATLAB handles array slicing:
"""
# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a.shape)   # Prints "(3,4) "

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the original array:
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape) # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)  # Prints "[[ 2]
                                             #          [ 6]
                                             #          [10]] (3, 1)"

"""
Integer array indexing: When you index into   arrays using slicing,
the resulting array view will always be a subarray of the original array.
 In contrast, integer array indexing allows you to construct arbitrary
 arrays using the data from another array. Here is an example:
"""
a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(a.shape)                    # prints "(3,2)"
print(a[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"
print(a[[0, 1, 2], [0, 1, 0]].shape)  # Prints "(3,)"
# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same element from the source array:
print(a[[0, 0], [1, 1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))  # Prints "[2 2]"


# One useful trick with integer array indexing is selecting or mutating one element from each row of a matrix:
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])  # Create a new array from which we will select elements
print(a)   # prints "array([[ 1,  2,  3],
              #                [ 4,  5,  6],
              #                [ 7,  8,  9],
             #                [10, 11, 12]])"

# Create an array of indices
b = np.array([0, 2, 0, 1])
print(b)
# Select one element from each row of a using the indices in b
print(np.arange(4))          # Prints "[0 1 2 3]"
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10
print(a)  # prints "array([[11,  2,  3],
                    #                [ 4,  5, 16],
                    #                [17,  8,  9],
                    #                [10, 21, 12]])


"""
Boolean array indexing: Boolean array indexing lets you pick out arbitrary elements of an array.
Frequently this type of indexing is used to select the elements of an array that satisfy some condition.
Here is an example:
"""
a = np.array([[1, 2], [3, 4], [5, 6]])

bool_idx = (a > 2)  # Find the elements of a that are bigger than 2;
# this returns a numpy array of Booleans of the same shape as a,
# where each slot of bool_idx tells whether that element of a is > 2.

print(bool_idx)  # Prints "[[False False]
                        #          [ True  True]
                        #          [ True  True]]"


# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values of bool_idx
print(a[bool_idx])  # Prints "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print(a[a > 2])  # Prints "[3 4 5 6]"

