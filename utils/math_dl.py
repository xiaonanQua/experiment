"""
收录深度学习中用的数学函数并可视化
"""
import numpy
import matplotlib.pyplot as plt
import math

def tanh(x, vis=False):
    """
    实现tanh函数=（exp(x)-exp(x)）/(exp(x)+exp(-x))
    :param x:输入的数据, list格式[x1,x2,x3]
    :param vis:是否可视化
    :return:
    """
    y = [(math.exp(x_)-math.exp(-x_))/(math.exp(x_)+math.exp(-x_))
         for x_ in x ]
    print('Input data2:{}, \nCompute Result:{}'.format(x, y))
    if vis:
        visual(x, y, 'tanh')

def logistic(x, vis=True):
    """
    实现sigmoid函数中的Logistic函数，log=1/(1+exp(-x))
    :param x:输入数据，list格式
    :param vis:是否可视化
    :return:
    """
    y = [1.0/(1+math.exp(-x_)) for x_ in x]
    print('Input data2:{}\n Compute Result:{}'.format(x, y))
    if vis:
        visual(x, y, 'Logistic')

def visual(x, y, name='math'):
    plt.plot(x, y, color='blue', marker='o')
    plt.title(name)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


if __name__ == '__main__':
    x = [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0,
         1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    tanh(x, vis=True)
    logistic(x, vis=True)
