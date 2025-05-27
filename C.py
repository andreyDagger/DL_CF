import sys
import threading
import numpy as np
from scipy.optimize import minimize

sys.setrecursionlimit(10**5+5)
threading.stack_size(2**26)


def main():
    n, m = map(int, input().split())
    k = n - m + 1
    a = [[0 for j in range(n)] for i in range(n)]
    c = [[0 for j in range(m)] for i in range(m)]
    for i in range(n):
        a[i] = list(map(int, input().split()))
    for i in range(m):
        c[i] = list(map(int, input().split()))
    A = np.zeros((m * m, k * k))
    B = np.zeros((m * m))
    for i in range(m):
        for j in range(m):
            B[i * m + j] = c[i][j]
            for wi in range(k):
                for wj in range(k):
                    A[i * m + j][wi * k + wj] += a[i + wi][j + wj]
    x, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
    for i in range(k):
        for j in range(k):
            print('{:.10f}'.format(x[i * k + j]), end=' ')
        print()


threading.Thread(target=main).start()
