"""
tuples
A tuple is an (immutable) ordered list of values. A tuple is in many ways similar to a list; one of the most
important differences is that tuples can be used as keys in dictionaries and as elements of sets, while
 lists cannot.
"""

d = {(x, x + 1): x for x in range(6)}  # Create a dictionary with tuple keys
print(d)         # print "{(0, 1): 0, (1, 2): 1, (5, 6): 5, (2, 3): 2, (4, 5): 4, (3, 4): 3}"
t = (5, 6)       # Create a tuple
print(type(t))    # Prints "<class 'tuple'>"
print(d[t])       # Prints "5"
print(d[(1, 2)])  # Prints "1"

