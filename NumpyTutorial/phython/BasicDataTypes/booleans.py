"""
Booleans: Python implements all of the usual operators for Boolean logic, but uses English words rather than symbols (&&, ||, etc.):
"""

t = True
f = False
print(type(t))    # Prints "<class 'bool'>"
print(t and f)    # Logical AND; prints "False"
print(t or f)      # Logical OR; prints "True"
print( not t)      # Logical NOT; prints "False"
print(t != f)       # Logical XOR; prints "True"
