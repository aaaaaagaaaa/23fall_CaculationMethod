# 第三次上机作业 自适应Simpson算法
# 被积函数\sqrt{x}\ln{x} 求积区间[0,1] 精度10^{-4}

import numpy as np

epsilon = 10 ** (-4)

def func1(x):
    if x != 0:
        return np.sqrt(x) * np.log(x)
    else:
        return 0

def simpson_int(func, a , b, h):
    '''返回函数func在[a,b]区间的simpson求积值'''
    return h/6 * (func(a) + 4*func((a+b)/2) + func(b))

# 使用递推函数求解：
def simpson_recursive(func, a, b, tol, s0, fa, fb, fc, depth, max_depth):

    h = (b - a) / 2
    fd = func(a + h/2)
    fe = func(b - h/2)
    mid = (a + b) / 2
    # 计算子区间的 Simpson 积分估计值
    s1 = simpson_int(func, a, a+h, h)
    s2 = simpson_int(func, a+h, b, h)
    s = s1 + s2

    # 检查误差
    if depth >= max_depth or abs(s - s0) <= 15 * tol:
        return s + (s - s0) / 15

    # 递归细分左右两个子区间
    s_left = simpson_recursive(func, a, mid, tol / 2, s1, fa, fc, fd, depth + 1, max_depth)
    s_right = simpson_recursive(func, mid, b, tol / 2, s2, fc, fb, fe, depth + 1, max_depth)

    return s_left + s_right

def adaptive_simpson(func, a, b, tol, max_depth=10):

    # 计算初始的 Simpson 积分估计值
    h = (b - a) / 2
    mid = (a + b) / 2
    fa, fb, fc = func(a), func(b), func(mid)
    s0 = h * (fa + 4 * fc + fb) / 3

    # 递归细分
    result = simpson_recursive(func, a, b, tol, s0, fa, fb, fc, 1, max_depth)

    return result


print(adaptive_simpson(func1, 0, 1, epsilon))