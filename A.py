import sys
import threading

sys.setrecursionlimit(10**5+5)
threading.stack_size(2**26)


def popcnt(x):
    res = 0
    while x:
        res += x % 2
        x //= 2
    return res


def main():
    m = int(input())
    vals = [0] * (2 ** m)
    ones = []
    zeros = []
    for i in range(2 ** m):
        vals[i] = int(input())
        if vals[i]:
            ones.append(i)
        else:
            zeros.append(i)
    if len(ones) <= 2 ** (m - 1):
        if len(ones) == 0:
            a = [
                [1 for i in range(m + 1)]
                ]
            b = [0, -1]
        else:
            a = [[0 for i in range(m + 1)] for j in range(len(ones))]
            for i in range(len(ones)):
                a[i][m] = -2*popcnt(ones[i])+1
                for j in range(m):
                    if ones[i] >> j & 1:
                        a[i][j] = 2
                    else:
                        a[i][j] = -100
            b = [0 for i in range(len(ones) + 1)]
            b[len(ones)] = -1
            for i in range(len(ones)):
                b[i] = 2
    else:
        if len(zeros) == 0:
            a = [
                [1 for i in range(m + 1)]
            ]
            b = [0, 1]
        else:
            a = [[0 for i in range(m + 1)] for j in range(len(zeros))]
            for i in range(len(zeros)):
                a[i][m] = -2 * popcnt(zeros[i]) + 1
                for j in range(m):
                    if not (zeros[i] >> j & 1):
                        a[i][j] = -100
                    else:
                        a[i][j] = 2
            b = [0 for i in range(len(zeros) + 1)]
            b[len(zeros)] = 1
            for i in range(len(zeros)):
                b[i] = -2
    print(2)
    print(len(a), 1)
    for i in range(len(a)):
        for j in range(len(a[i])):
            print(a[i][j], end=' ')
        print()
    for i in range(len(b)):
        print(b[i], end=' ')


threading.Thread(target=main).start()
