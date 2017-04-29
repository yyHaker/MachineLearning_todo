"""
integers, floats

note: 1.unlike many languages, Python does not have unary increment (x++) or decrement (x--) operators.
        2.Python also has built-in types for long integers and complex numbers;
"""

x = 3
print(type(x))  # prints "<class 'int'>"
print(x)           # prints '3'
print(x+1)        # Addition; prints "4"
print(x-1)        # Subtraction; prints "2"
print(x*2)       # Multiplication; prints "6"
print(x ** 2)   # Exponentiation; prints "9"
x += 1
print(x)           # Prints "4"
x *= 2
print(x)           # Prints "8"
y = 2.5
print(type(y))  # prints "<class 'float'>"
print(y, y + 1, y*2, y ** 2)  # Prints "2.5 3.5 5.0 6.25"
