# 简化Newton法求根

import numpy as np
import sympy as sp

def f(x):
    return x ** 3 - 5 * x + 4.272

def f_prime(x_value):
    x = sp.symbols('x')
    f = x ** 3 - 5 * x + 4.272

    return sp.lambdify(x, f.diff(x), 'numpy')(x_value)


def newton_simplified_method(f, x_0, N=10, epslion=10**(-5)):
    '''牛顿迭代法求根'''
    x = np.zeros(N)
    f_x = np.zeros(N)
    f_prime_f = np.zeros(N)
    x[0] = x_0
    f_prime_0 = f_prime(x_0)
    
    for i in range(N-1):
    
        f_x[i] = f(x[i])
        f_prime_f[i] = f_x[i] / f_prime_0
        x[i+1] = x[i] - f_x[i] / f_prime_0

        if abs(x[i+1] - x[i]) < epslion:
            break
    
    return {'x':x, 'f_x':f_x, 'f_prime':f_prime_0 ,'f_prime_f':f_prime_f}

print(newton_simplified_method(f, 1.15))