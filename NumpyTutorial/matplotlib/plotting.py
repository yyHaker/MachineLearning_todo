"""
Matplotlib is a plotting library. In this section give a brief introduction
to the matplotlib.pyplot module, which provides a plotting system similar
to that of MATLAB.
see more at :http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
"""

# The most important function in matplotlib is plot, which allows you to plot 2D data.
# Here is a simple example:
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on a sine curve
# x = np.arange(0, 3 * np.pi, 0.1)
# y = np.sin(x)

# Plot the points using matplotlib
# plt.plot(x, y)
# plt.show()  # You must call plt.show() to make graphics appear.


# With just a little bit of extra work we can easily plot multiple lines at once,
# and add a title, legend, and axis labels:
# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()
