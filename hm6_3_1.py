# 迭代法计算非线性方程的根

import numpy as np

def iteration_function(x):
    return np.array([1/6 + 1/3 * np.cos(x[1] * x[2]),
                     1/9 * np.sqrt(x[0] ** 2 + np.sin(x[2]) + 1.06) -0.1,
                     -1/20 * np.exp(- x[0] * x[1]) - 1/20 * (10 * np.pi/3 -1)])

def iteration_method(phi, x_0=np.zeros(3), N=100, epslion=10**(-5)):
    for i in range(N):
        x_1 = phi(x_0)
        if np.linalg.norm(x_1 - x_0) < epslion:
            break
        x_0 = x_1

    return x_1

print(iteration_method(iteration_function))