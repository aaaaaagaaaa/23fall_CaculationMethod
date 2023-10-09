#第一次作业T2

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.interpolate import CubicSpline

# data given by the exercise
x_value = np.array([0, 1, 4, 9, 16, 25, 36, 49, 64])
y_value = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])



def get_lag_func(value_x, value_y):
    x = sp.symbols('x')

    def lagrange_base(i, value = x_value):
        x_new = np.concatenate((value[:i], value[i+1:]))  # remove the element of the order i
        
        numera = np.prod(x - x_new)
        denomi = np.prod(value[i] - x_new)
        return numera/denomi
    
    index_list = np.arange(len(value_x))
    bases = list(map(lagrange_base, index_list))
    f = np.dot(value_y, bases)
    print(sp.expand(f))
    return sp.lambdify(x, f, 'numpy')

def show_plot():
    plt.figure(figsize=(10,8))
    
    plt.subplot(2, 1, 1)
    x_v1 = np.linspace(0, 64, 100)
    y_v11 = np.sqrt(x_v1)
    y_v12 = get_lag_func(x_value, y_value)(x_v1)
    y_v13 = CubicSpline(x_value, y_value)(x_v1)
    plt.plot(x_v1, y_v11, c='b', label='original function')
    plt.plot(x_v1, y_v12, c='y', label='lagerange interpolating function')
    plt.plot(x_v1, y_v13, c='g', label='cubic spline function')
    plt.title('interval [0,64]')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    x_v2 = np.linspace(0, 1, 100)
    y_v21 = np.sqrt(x_v2)
    y_v22 = get_lag_func(x_value, y_value)(x_v2)
    y_v23 = CubicSpline(x_value, y_value)(x_v2)
    plt.plot(x_v2, y_v21, c='b', label='original function')
    plt.plot(x_v2, y_v22, c='y', label='lagerange interpolating function')
    plt.plot(x_v2, y_v23, c='g', label='cubic spline function')
    plt.title('interval [0,1]')
    plt.legend()
    
    plt.show()

def cs_expression(x,y) -> list:
    '''get the expression of cubic spline function,which is a segmentation function'''
    coef = CubicSpline(x, y).c
    def coef_to_expression(coefficients):
        x = sp.symbols('x')
        poly = []
        for j in range(len(coefficients[0])):
            # cubic spline function has degree of 3, i.e. len(coefficients) == 4
            poly.append(coefficients[0][j]*x**3+coefficients[1][j]*x**2+coefficients[2][j]*x+coefficients[3][j])
        return poly
    return coef_to_expression(coef)

# print(cs_expression(x_value, y_value))

# xx = np.linspace(0, 10, 10)
# yy = np.sin(xx)

# cs2 = CubicSpline(xx, yy)
# x_1 = np.linspace(0, 10, 1000)
# y_1 = cs2(x_1)
# plt.plot(x_1, y_1, c='y', label='cs func')
# plt.plot(x_1, np.sin(x_1), c='r', label='sin func')
# plt.legend()
# plt.show()

