import numpy as np

def r2c_7(a):   # 映射到MOSEI的七分类
    if a < -2:
        res = 0
    elif a < -1:
        res = 1
    elif a < 0:
        res = 2
    elif a <= 0:
        res = 3
    elif a <= 1:
        res = 4
    elif a <= 2:
        res = 5
    elif a > 2:
        res = 6
    else:
        print('result can not be transferred to 7-class label', a)  # 示例输出：result can not be transferred to 7-class label: -5.2
        raise NotImplementedError   # 异常处理提示
    return res

def mosi_r2c_7(a):  # 映射MOSI的七分类
    return np.int64(np.round(a)) + 3    # 对a四舍五入加3后转换为64位整数，与MOSI数据集对应。

def r2c_2(a):   # 映射MOSI和MOSEI的二分类
    if a < 0:
        res = 0
    else:
        res = 1
    return res

def pom_r2c_7(a):   # 映射到pom的七分类
    # [1,7] => 7-class
    if a < 2:
        res = -3
    if 2 <= a and a < 3:
        res = -2
    if 3 <= a and a < 4:
        res = -1
    if 4 <= a and a < 5:
        res = 0
    if 5 <= a and a < 6:
        res = 1
    if 6 <= a and a < 7:
        res = 2
    if a >= 7:
        res = 3
    return res + 3
