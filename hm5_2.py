# 五点差分法
import numpy as np

N = 10 # 分点数

# 设置矩阵A
A = np.zeros([N**2, N**2])
for i in range(N):
    for j in range(N):
        A[N*i+j, N*i+j] = -4 # 中心格点
        if i != 0:
            A[N*i+j, N*(i-1)+j] = 1# 上格点
        if j != 0:
            A[N*i+j, N*i+j-1] = 1# 左格点
        if j != 9:
            A[N*i+j, N*i+j+1] = 1# 右格点
        if i != 9:
            A[N*i+j, N*(i+1)+j] = 1# 下格点

# 设置方程常值向量
b = np.array([(i**2 + j**2) * np.e ** (i*j/100) for i in range(1, N+1) for j in range(1, N+1)])

def exact_solution():
    
    def f(i, j):
        return np.e ** (i * j)
   
    u_exact = [f(i/10, j/10) for i in range(1, N+1) for j in range(1, N+1)]

    return u_exact

def Jacobi(A, b):
    D = np.diag(np.diag(A))
    L_U = D - A # L+U
    
    u_1 = np.array([1]*(N**2)) #设置迭代的初始值
    ind = 0
    while ind < 1000: # 最大迭代次数1000
        u_0 = u_1
        u_1 = np.linalg.inv(D) @ L_U @ u_1 + np.linalg.inv(D) @ b
        ind += 1
        if np.linalg.norm(u_1 - u_0, ord=np.inf) < 10 ** (-5):
            break
    
    return {'u':np.round(u_1, decimals=4) , 'ind':ind}

def SOR(A, b, omega):
    D = np.diag(np.diag(A))
    L = -np.tril(A, k=-1)
    U = D - L - A

    B = np.linalg.inv(D) @ (D - A)
    L_omega = np.linalg.inv(D - omega * L) @ ((1 - omega) * D + omega * U)

    u_1 = np.array([1]*(N**2))
    ind = 0
    while ind < 1000:
        u_0 = u_1
        u_1 = L_omega @ u_1 + omega * np.linalg.inv(D - omega * L) @ b
        ind += 1
        if np.linalg.norm(u_1 - u_0, ord=np.inf) < 10 ** (-5):
            break
    
    return {'u':np.round(u_1, decimals=4), 'ind':ind}

import numpy as np

def C_G(A, b):
    n = len(b)
    
    x0 = np.array([1]*(N**2))

    r = b - A @ x0
    p = r
    rsold = np.dot(r, r)

    ind = 0
    while ind < 1000:
        Ap = A @ p
        alpha = rsold / np.dot(p, Ap)
        x0 = x0 + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)
        ind += 1

        if np.sqrt(rsnew) < 10 ** (-5):
            break

        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    return {'x':np.round(x0, decimals=4), 'ind':ind}


# print(Jacobi(A, b)['u'])
# print(SOR(A, b, 1)['u'])
# print(SOR(A, b, 1.25)['u'])
print(SOR(A, b, 1.5)['u'])
# print(SOR(A, b, 1.75)['u'])
# print(C_G(A, b)['x'].tolist())
# print(A)


        