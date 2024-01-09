# 非线性方程组的Newton迭代法
import numpy as np


def f(x):
    return np.array([3 * x[0] - np.cos(x[1] * x[2]) - 1/2,
                     x[0] ** 2 - 81 * (x[1] + 0.1) ** 2 + np.sin(x[2]) + 1.06,
                     np.exp(-x[0] * x[1]) + 20 * x[2] + 10*np.pi / 3 - 1])


def f_prime(x):
    return np.array([[3, x[2] * np.sin(x[1]*x[2]), x[1] * np.sin(x[1]*x[2])],
                     [2 * x[0], -162 * (x[1] + 0.1), np.cos(x[2])],
                     [-x[1] * np.exp(-x[0] * x[1]), -x[0] * np.exp(-x[0] * x[1]), 20]])

def Newton_iter(f, f_prime, x_0, N=1000, epslion=10**(-8)):
    for i in range(N):
        if np.linalg.det(f_prime(x_0)) == 0:
            print(f"矩阵奇异，迭代结束,迭代次数{i},前后项差{np.linalg.norm(x_1 - x_0)}")
            break
        x_1 = x_0 - np.linalg.inv(f_prime(x_0)) @ f(x_0)
        if np.linalg.norm(x_1 - x_0) < epslion:
            break
        x_0 = x_1

    return x_1

print(Newton_iter(f, f_prime, np.array([0.5, 0.5, 0.5])))
print(Newton_iter(f, f_prime, np.array([0.5, 0.2, 0.8])))
print(Newton_iter(f, f_prime, np.array([0.1, 0.3, 0.3])))
