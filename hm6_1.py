# 第六次作业 newton法求根

import numpy as np
import sympy as sp

def f(x):
    return x ** 3 - 5 * x + 4.272

def f_prime(x_value):
    x = sp.symbols('x')
    f = x ** 3 - 5 * x + 4.272

    return sp.lambdify(x, f.diff(x), 'numpy')(x_value)


def newton_root_method(f, x_0, N=10, epslion=10**(-5)):
    '''牛顿迭代法求根'''
    key = 1
    x = np.zeros(N)
    f_x = np.zeros(N)
    f_prime_x = np.zeros(N)
    f_prime_f = np.zeros(N)
    x[0] = x_0
    
    for i in range(N-1):
        if f_prime(x[i]) != 0:
            f_x[i] = f(x[i])
            f_prime_x[i] = f_prime(x[i])
            f_prime_f[i] = f_x[i] / f_prime_x[i]

            x[i+1] = x[i] - f_x[i] / f_prime_x[i]

            if abs(x[i+1] - x[i]) < epslion:
                break
        else:
            key = 0
            break
    
    if key == 1:
        return {'x':x, 'f_x':f_x, 'f_prime':f_prime_x, 'f_prime_f':f_prime_f}
    else:
        print("Newton迭代求根法失效")

print(newton_root_method(f, 1.15))
    