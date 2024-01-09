# 第二题 估计矩阵1-范数

import numpy as np

def estimate_norm1(B:np.array):
    '''优化法估计矩阵B的1范数'''
    n = B.shape[0]
    x = np.array([1/n]*n)

    k = 1
    while k == 1:
        w = B @ x
        v = [i/np.abs(i) for i in w]
        z = np.transpose(B) @ v
        z_abs = np.abs(z)

        if np.max(z_abs) <= np.transpose(z) @ x:
            k = 0
            return np.sum(w)
        else:
            j = np.argmax(z_abs)
            x = np.zeros(n)
            x[j] = 1

def Hilbert_cond(n):
    '''传入Hilbert矩阵阶数'''
    H = np.array([[1/(i + j - 1) for j in range(1, n+1)] for i in range(1, n+1)])
    H_inv = np.linalg.inv(H)
    return np.linalg.norm(H, ord=np.inf) * np.linalg.norm(H_inv, ord=np.inf)

def gauss_elimination(A, b):
    n = len(b)
    augmented_matrix = np.column_stack((A, b))

    for i in range(n-1):
        max_row = np.argmax(np.abs(augmented_matrix[i:, i])) + i
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]

        # 消元过程
        for j in range(i+1, n):
            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]

    # 回代求解
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i+1:n], x[i+1:])) / augmented_matrix[i, i]

    return x

def construct_A(n):
    '''传入矩阵阶数n, 构建A_n'''
    A = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if j == i:
                A[i][j] = 1
            elif j < i :
                A[i][j] = -1
            elif j == n-1:
                A[i][j] = 1
    return A

dim = 5
while dim <= 30:
    A_5 = construct_A(dim)
    x = np.random.rand(dim) # 生成随机向量
    b = A_5 @ x
    xdeltax = gauss_elimination(A_5, b)

    # print(xdeltax - x)
    print(np.linalg.norm(xdeltax - x)/np.linalg.norm(x))
    dim += 5
# print(Hilbert_cond(20))