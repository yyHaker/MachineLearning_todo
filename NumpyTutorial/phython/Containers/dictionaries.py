"""
Dictionaries
A dictionary stores (key, value) pairs, similar to a Map in Java or an object in Javascript.
"""

d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(d['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in d)     # Check if a dictionary has a given key; prints "True"
d['fish'] = 'wet'    # Set an entry in a dictionary
print(d['fish'])      # Prints "wet"
# print d['monkey']  # KeyError: 'monkey' not a key of d
print(d.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(d.get('fish', 'N/A'))   # Get an element with a default; prints "wet"
del d['fish']        # Remove an element from a dictionary
print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"


# Loops: It is easy to iterate over the keys in a dictionary:
d = {'person': 2, 'car': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs)) # Prints "A person has 2 legs", "A spider has 8 legs", "A cat has 4 legs"

# If you want access to keys and their corresponding values, use the iteritems method:
d = {'person': 2, 'car': 4, 'spider': 8}
# for animal, legs in d.iteritems():
   #  print('A %s has %d legs' % (animal, legs))
# Prints "A person has 2 legs", "A spider has 8 legs", "A cat has 4 legs"

# Dictionary comprehensions: These are similar to list comprehensions, but allow you to easily construct dictionaries.
nums =[0, 1, 2, 3, 4]
even_num_to_square ={x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)

