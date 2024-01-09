# 计算方法第四次上机作业第一题

import numpy as np

A = np.array([[10, 7, 8, 7], 
              [7, 5, 6, 5],
              [8, 6, 10, 9],
              [7, 5, 9, 10]])

b = np.array([32, 23, 33, 31])

A_delta = np.array([[10, 7, 8.1, 7.2], 
                    [7.08, 5.04, 6, 5],
                    [8, 5.98, 9.89, 9],
                    [6.99, 5, 9, 9.98]])

det_A = np.linalg.det(A) # 矩阵行列式
lambdas = np.linalg.eigvals(A) # 矩阵特征值
cond_A = np.linalg.cond(A) # 计算cond(A)

#求解方程Ax=b和方程(A+\deltaA)x = b:
x = np.linalg.solve(A, b)
xdeltax = np.linalg.solve(A_delta, b)

delta_x = xdeltax - x # 得到向量\delta x
delta_x_norm2 = np.linalg.norm(delta_x) # 得到\delta x 的2范数

error_x = np.linalg.norm(x) / delta_x_norm2
error_A = np.linalg.norm(A_delta - A, ord=2) / np.linalg.norm(A, ord=2)

print(delta_x)
print(delta_x_norm2)
print(error_x)
print(error_A)
print(np.linalg.norm(np.linalg.inv(A),ord=2) * np.linalg.norm(A, ord=2))