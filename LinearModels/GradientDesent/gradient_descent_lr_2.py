"""
梯度下降法实现线性回归
y=mx+b
m is the slope, b is y-intercept
具体思路方法查看：https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/
"""
from numpy import *
import matplotlib.pyplot as plt


# 定义cost function
def compute_error_for_line_given_points(b, m, points):
    totalError=0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y-(m*x+b))**2
        return totalError/float(len(points))


# 梯度每下降一步，计算更新相应的值
def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N)*(y-(x*m_current+b_current))
        m_gradient += -(2/N)*x*(y-(x*m_current+b_current))
    new_b = b_current-(learningRate * b_gradient)
    new_m = m_current-(learningRate * m_gradient)
    return [new_b, new_m]


# 梯度下降求解模型y=m*x+b
def gradient_descent_runner(points, starting_b, starting_m, learningRate, num_iterations):
    b = starting_b
    m =starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, points, learningRate)
    return [b, m]


def run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0003
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    print("Starting gradient decent at b = {0}, m = {1}, error ={2}".format(initial_b,  initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

    # 瞄点
    X = []
    Y = []
    Z = []   # 预测值
    for i in range(len(points)):
        X.append(points[i, 0])
        Y.append(points[i, 1])
        Z.append(m*points[i, 0]+b)
    plt.scatter(X, Y)
    plt.plot(X, Z)
    plt.show()


if __name__ == '__main__':
    run()
