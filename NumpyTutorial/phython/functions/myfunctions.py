"""
functions
"""


def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))   # Prints "negative", "zero", "positive"


# define functions to take optional keyword arguments
def hello(name, loud=False):
    if loud:
        print('Hello,%s' % name.upper())
    else:
        print('hello,%s' % name)

hello('Bob')  # Prints "Hello, Bob"
hello('Fred', loud=True)  # Prints "HELLO, FRED!"

