# 计算方法第二次作业T1
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import minimize

#data set
x = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
y = [1.0, 0.41, 0.50, 0.61, 0.91, 2.02, 2.46]

def get_ploy_coeff(x, y, times):
    return np.polyfit(x, y, times)

def get_ploy_func(x, y, times):
    return np.poly1d(get_ploy_coeff(x, y, times))

# print(np.round(get_ploy_coeff(x, y, 3), decimals=4))

def draw_pic_poly():
    '''画出拟合的三次三次多项式的图像，在一张图上实现 '''
    f3 = get_ploy_func(x, y, 3)
    f4 = get_ploy_func(x, y, 4)

    x_data = np.linspace(0, 1, 100)
    y_3 = f3(x_data)
    y_4 = f4(x_data)

    plt.plot(x_data, y_3, c='r', label='cubic polynomial')
    plt.plot(x_data, y_4, c='b', label='quaraic polynomial')
    plt.legend()
    plt.show()

# draw_pic()

#求正弦型拟合函数
def imitate_func(para):
    '''x为三元数组 para=[k, \theta, A]为拟合函数的三个参数, 返回函数'''
   
    def func(x):
        return np.sin(para[0] * x + para[1]) + para[2]
    
    return func

def residual_sum(para):
    '''残差的平方和函数'''
    return np.sum([(imitate_func(para)(x_i) - y_i) ** 2 for (x_i, y_i) in zip(x, y) ])


def get_sine_coeff(x, y):
    initial_guess = [1, 0, 0]

    result = minimize(residual_sum, initial_guess, method='BFGS')

    mini_para = result.x
    return mini_para

# print(np.round(get_sine_coeff(x, y), decimals=4))

def draw_pic_all():
    '''画出所有你和函数的图像，在一张图上实现 '''
    f3 = get_ploy_func(x, y, 3)
    f4 = get_ploy_func(x, y, 4)
    f_s = imitate_func(get_sine_coeff(x, y))
    

    x_data = np.linspace(0, 1, 100)
    y_3 = f3(x_data)
    y_4 = f4(x_data)
    y_s = f_s(x_data)

    plt.plot(x_data, y_3, c='r', label='cubic polynomial')
    plt.plot(x_data, y_4, c='b', label='quaraic polynomial')
    plt.plot(x_data, y_s, c='y', label='sine function')
    plt.legend()
    plt.show()

draw_pic_all()