# 求解线性方程组方法 Jacobi G-S SOR迭代法

import numpy as np

#设定常量的值
epslion = 1
a = 1/2
n = 100
h = 1/n
b = np.array([a * h **2] * n)
b[-1] = -1
N = 1000 # 设置迭代次数

#设定矩阵A
A = np.zeros((n,n))
A[0][0] = A[-1][-1] = -(2 * epslion +h)
A[0][1] = epslion + h
A[-1][-2] = epslion
for i in range(1, n-1):
    A[i][i] = -(2 * epslion +h)
    A[i][i-1] = epslion
    A[i][i+1] = epslion +h

def exact_solution():
    
    def f(x):
        return (1 - a) / (1 - np.e ** (-1 / epslion)) * (1 - np.e ** (-x / epslion)) + a * x
   
    x_list = np.arange(0, 1, 0.01)

    return f(x_list)

def Jacobi(A, b):
    D = np.diag(np.diag(A))
    L_U = D - A # L+U
    
    y = np.linspace(0, 1, 100) #设置迭代的初始值
    for k in range(N):
        y = np.linalg.inv(D) @ L_U @ y + np.linalg.inv(D) @ b 
    
    return np.round(y, decimals=4)

def G_S(A, b):
    D = np.diag(np.diag(A))
    L = -np.tril(A, k=-1)
    U = D - L - A

    y = np.array([1]*100)
    for k in range(N):
        y = np.linalg.inv(D - L) @ U @ y + np.linalg.inv(D - L) @ b
    
    return np.round(y, decimals=4)

def SOR():
    D = np.diag(np.diag(A))
    L = -np.tril(A, k=-1)
    U = D - L - A

    B = np.linalg.inv(D) @ (D - A)
    omega = 2 / (1 + np.sqrt(1 - np.linalg.norm(B, ord=2)**2))# 最佳松弛因子
    L_omega = np.linalg.inv(D - omega * L) @ ((1 - omega) * D + omega * U)

    y = np.arange(0, 1, 0.01)
    for k in range(N):
        y = L_omega @ y + omega * np.linalg.inv(D - omega * L) @ b
    
    return np.round(y, decimals=4)

# print(Jacobi(A, b).tolist())
# print(G_S(A, b).tolist())
# print(SOR().tolist())
print(np.linalg.norm(Jacobi(A, b) - exact_solution(), ord=2))
print(np.linalg.norm(SOR() - exact_solution(), ord=2))
print(np.linalg.norm(G_S(A, b) - exact_solution(), ord=2))
# print(exact_solution())
# print(A)
# print(b)