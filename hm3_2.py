# 计算方法上机题T1问题2问题三
# Romberg求积算法  自适应算法
# 被积函数为\sqrt{x}\ln{x} 积分区间[0,1]

import numpy as np

epsilon = 10 ** (-6) #计算精度
N = 20 #计算romberg算法递推N次
T = [[0 for i in range(N)] for j in range(N)] #二维列表储存计算所得的数据 第ij个元素即为T_j^{(i)} 初始化为10*10的空数组

def func1(x):
    if x != 0:
        return np.sqrt(x) * np.log(x)
    if x == 0:
        return 0


def romberg(func):

    T[0][0] = 1/2 * (func(0) + func(1))

    for k in range(1,N):
        # 计算第k行的值
        h = 1 / (2**k)
        x_value = np.array([i*h for i in range(2 ** k + 1)])
        y_value = [func(xi) for xi in x_value]
        T[k][0] = 1/2 * T[k-1][0] + h * np.sum(y_value[1:-1:2])

        for j in range(1, k+1):
            T[k-j][j] = 4**j / (4**j-1) * T[k-j+1][j-1] - 1/(4**j - 1) * T[k-j][j-1]
        
        if np.abs(T[0][k] - T[0][k-1]) < epsilon:
            break

    print(T)
    print(k)
    return T[0][k]

print(T[0][0])
print(romberg(func1))
