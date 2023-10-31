# 计算方法第三次作业 数值积分
# 本程序计算第一题(1)以及第二题

import numpy as np
from scipy import integrate

def func1(x):
    if x != 0:
        return np.sqrt(x) * np.log(x)
    if x == 0:
        return 0

def func2(x):
    return np.exp(x)

def recom_trap(h, func):
    '''复合梯形公式求解数值积分'''
    fx_i = [func(i * h) for i in range(int(1/h) + 1)]
    T_h = h/2 * (fx_i[0] + fx_i[-1] + 2 * np.sum(fx_i[1:-1]))
    return T_h

def recom_simpson(h, func):
    '''复合辛普森公式求解数值积分'''
    fxi = [func(i * h/2) for i in range(int(2/h)+1)]
    S_h = h/6 * (fxi[0] + fxi[-1] + 4 * np.sum(fxi[1:-1:2]) + 2 * np.sum(fxi[2:-2:2]))
    return S_h

def Newton_Cotes(n, func):
    '''返回关于函数f的n次复合N-C在[0,1]求积公式的值'''
    n = 2 
    h = 1/n
    x_value = np.array([i * h for i in range(n+1)])
    y_value = func(x_value)
    
    def n_c_func(k):
        '''返回一个求得C_k的被积函数'''
        def cotes_func(t):
            res = 1
            for j in range(n+1):
                if j != k:
                    res *= (t-j)
            return res
        return cotes_func
    
    C_n = np.array([(-1)**(n-k)/(n * np.math.factorial(k) * np.math.factorial(n-k))* integrate.quad(n_c_func(k), 0, n)[0] for k in range(n+1)])
   
    return np.dot(y_value, C_n)

def Romberg():
    
    
    return 0

#第一题的输出数据
# nums = np.array([1, 0.1, 0.01, 0.001, 0.0001])
# trap_value = [recom_trap(i, func1) for i in nums]
# simpson_value = [recom_simpson(i, func1) for i in nums]
# print(np.round(trap_value, decimals=6))
# print(np.round(simpson_value, decimals=6))

#第二题输出数据
nums2 = np.array([1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128])
n = [2, 4, 8, 16, 32, 64, 128]
trap_value2 = [recom_trap(i, func2) for i in nums2]
simpson_value2 = [recom_simpson(i, func2) for i in nums2]
n_c = [Newton_Cotes(i, func2) for i in n]
# print(trap_value2)
# print(simpson_value2)
print(n_c)



