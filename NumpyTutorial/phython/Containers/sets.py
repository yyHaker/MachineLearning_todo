"""
sets
A set is an unordered collection of distinct elements.
"""

animals = {'cat', 'dog'}
print('cat' in animals)   # Check if an element is in a set; prints "True"
print('fish' in animals)  # prints "False"
animals.add('fish')      # Add an element to a set
print('fish' in animals)  # Prints "True"
print(len(animals))       # Number of elements in a set; prints "3"
animals.add('cat')       # Adding an element that is already in the set does nothing
print(len(animals))       # Prints "3"
animals.remove('cat')    # Remove an element from a set
print(len(animals))       # Prints "2"

# Loops: Iterating over a set has the same syntax as iterating over a list;
# however since sets are unordered, you cannot make assumptions about
#  the order in which you visit the elements of the set:
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))  # Prints "#1: fish", "#2: dog", "#3: cat"

# Set comprehensions: Like lists and dictionaries, we can easily construct sets using set comprehensions:
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print(nums)  # Prints "{0, 1, 2, 3, 4, 5}"
