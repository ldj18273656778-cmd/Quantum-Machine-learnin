import numpy as np
from pathlib import Path

def encode_diagonal(k, n=10):
    """
    生成一个 n×n 的对角线编码矩阵（flatten 后就是 n^2 = 100 bit）
    
    参数：
        k : int (0~n-1)，表示数字
        n : 矩阵大小（默认10）
    
    返回：
        shape = (n, n) 的 numpy array
    """
    M = np.zeros((n, n), dtype=int)
    for i in range(n):
        j = (i + k) % n
        M[i, j] = 1
    return M

if __name__ == "__main__":
    a=encode_diagonal(1)
    b=np.array([2,3,0])
    b=np.array([encode_diagonal(b[x]) for x in range(len(b))])#针对形状为(N,)的array;就得利用循环
    print(b.shape)
    print(b[1])