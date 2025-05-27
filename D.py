import sys
import threading
import numpy as np

# sys.setrecursionlimit(10**5+5)
# threading.stack_size(2**26)


def read3d(d, r, c, vals):
    idx = 0
    a = np.zeros((d, r, c))
    for d in range(d):
        for i in range(r):
            for j in range(c):
                a[d][i][j] = vals[idx]
                idx += 1
    return a


def do_relu(mat, alpha):
    res = np.zeros(mat.shape)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            for k in range(mat.shape[2]):
                if mat[i][j][k] > 0:
                    res[i][j][k] = mat[i][j][k]
                else:
                    res[i][j][k] = mat[i][j][k] * alpha
    return res


def do_pool(mat, s):
    res = np.zeros((mat.shape[0], mat.shape[1] // s, mat.shape[2] // s))
    for d in range(res.shape[0]):
        for i in range(res.shape[1]):
            for j in range(res.shape[2]):
                mx = -np.inf
                for ii in range(i * s, (i + 1) * s):
                    for jj in range(j * s, (j + 1) * s):
                        mx = max(mx, mat[d][ii][jj])
                res[d][i][j] = mx
    return res


def do_bias(mat, b):
    res = np.zeros(mat.shape)
    for d in range(mat.shape[0]):
        for i in range(mat.shape[1]):
            for j in range(mat.shape[2]):
                res[d][i][j] = mat[d][i][j] + b[d]
    return res


def do_cnv(mat, mode, h, k, s, p, a):
    d = mat.shape[0]
    a = np.array(a).reshape((h, d, k, k))
    new_size = (mat.shape[1] - k + 2 * p) // s + 1
    res = np.zeros((h, new_size, new_size))
    if mode == 'm':
        padded = np.pad(mat, ((0, 0), (p, p), (p, p)), 'reflect')
    elif mode == 'e':
        padded = np.pad(mat, ((0, 0), (p, p), (p, p)), 'edge')
    else:
        padded = np.pad(mat, ((0, 0), (p, p), (p, p)), 'wrap')
    for t in range(h):
        for i in range(new_size):
            for j in range(new_size):
                for w in range(d):
                    for ki in range(k):
                        for kj in range(k):
                            res[t][i][j] += padded[w][i*s+ki][j*s+kj]*a[t][w][ki][kj]
    return res


def back_relu(grad, alpha, mat_in):
    res = np.zeros(grad.shape)
    for t in range(res.shape[0]):
        for i in range(res.shape[1]):
            for j in range(res.shape[2]):
                res[t][i][j] = grad[t][i][j]
                if mat_in[t][i][j] < 0:
                    res[t][i][j] *= alpha
    return res, None


def back_pool(grad, s, mat_in):
    res = np.zeros(mat_in.shape)

    for t in range(grad.shape[0]):
        for i in range(grad.shape[1]):
            for j in range(grad.shape[2]):
                mx = -np.inf
                for wi in range(i * s, (i + 1) * s):
                    for wj in range(j * s, (j + 1) * s):
                        mx = max(mx, mat_in[t][wi][wj])
                for wi in range(i * s, (i + 1) * s):
                    for wj in range(j * s, (j + 1) * s):
                        if mat_in[t][wi][wj] == mx:
                            res[t][wi][wj] += grad[t][i][j]
    return res, None


def back_bias(grad, b, mat_in):
    return grad.copy(), np.array([np.sum(grad[d]) for d in range(len(b))])


def get_alt_1d(i, n, mode, p):
    assert p < n
    if mode == 'm':
        if i < p:
            i = p + (p - i)
        elif i >= n - p:
            i = (n - p - 1) - (i - (n - p - 1))
    elif mode == 'e':
        if i < p:
            i = p
        elif i >= n - p:
            i = n - p - 1
    else:
        if i < p:
            i = n - p - 1 - (p - i - 1)
        elif i >= n - p:
            i = p + (i - (n - p))
    return i


def get_alt(i, j, n, m, mode, p):
    return get_alt_1d(i, n, mode, p), get_alt_1d(j, m, mode, p)


def back_cnv(grad, mode, h, k, s, p, a, mat_in):
    d = mat_in.shape[0]
    a = np.array(a).reshape((h, d, k, k))
    grad_kernel = np.zeros_like(a)

    if mode == 'm':
        padded = np.pad(mat_in, ((0, 0), (p, p), (p, p)), 'reflect')
    elif mode == 'e':
        padded = np.pad(mat_in, ((0, 0), (p, p), (p, p)), 'edge')
    elif mode == 'c':
        padded = np.pad(mat_in, ((0, 0), (p, p), (p, p)), 'wrap')
    res = np.zeros_like(padded)

    for t in range(h):
        for i in range(grad.shape[1]):
            for j in range(grad.shape[2]):
                for w in range(d):
                    for ki in range(k):
                        for kj in range(k):
                            res[w][i*s+ki][j*s+kj] += grad[t][i][j] * a[t][w][ki][kj]
                            grad_kernel[t][w][ki][kj] += grad[t][i][j] * padded[w][i*s+ki][j*s+kj]
    if p > 0:
        for t in range(res.shape[0]):
            for i in range(res.shape[1]):
                for j in range(res.shape[2]):
                    i1, j1 = get_alt(i, j, res.shape[1], res.shape[2], mode, p)
                    if i == i1 and j == j1:
                        continue
                    res[t][i1][j1] += res[t][i][j]
        res = res[:, p:-p, p:-p]
    return res, grad_kernel


def print_3d(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            for k in range(mat.shape[2]):
                print('{:.6f}'.format(mat[i][j][k]), end=' ')
    print()


def main():
    qwe = input().split()
    n0, d0 = int(qwe[0]), int(qwe[1])
    mat = read3d(d0, n0, n0, list(map(int, qwe[2:])))
    init = mat.copy()
    l = int(input())
    op = ['' for i in range(l)]
    info = [None] * l
    out = [None for i in range(l)]
    for i in range(l):
        s = input().split()
        op[i] = s[0]
        if op[i] == 'relu':
            info[i] = 1/float(s[1])
        elif op[i] == 'pool':
            info[i] = int(s[1])
        elif op[i] == 'bias':
            info[i] = list(map(int, s[1:]))
        else:
            assert op[i][:3] == 'cnv'
            info[i] = list(map(int, s[1:]))
    for i in range(l):
        if op[i] == 'relu':
            mat = do_relu(mat, info[i])
        elif op[i] == 'pool':
            mat = do_pool(mat, info[i])
        elif op[i] == 'bias':
            mat = do_bias(mat, info[i])
        else:
            mat = do_cnv(mat, op[i][3], info[i][0], info[i][1], info[i][2], info[i][3], info[i][4:])
        out[i] = mat
    vals = list(map(int, input().split()))
    grad = read3d(mat.shape[0], mat.shape[1], mat.shape[2], vals)
    grad_param = [None] * l
    for i in reversed(range(l)):
        mat_in = init
        if i > 0:
            mat_in = out[i - 1]
        if op[i] == 'relu':
            grad, gp = back_relu(grad, info[i], mat_in)
        elif op[i] == 'pool':
            grad, gp = back_pool(grad, info[i], mat_in)
        elif op[i] == 'bias':
            grad, gp = back_bias(grad, info[i], mat_in)
        else:
            grad, gp = back_cnv(grad, op[i][3], info[i][0], info[i][1], info[i][2], info[i][3], info[i][4:], mat_in)
        grad_param[i] = gp
    print_3d(mat)
    print_3d(grad)
    for i in range(l):
        if op[i] == 'bias':
            for x in grad_param[i]:
                print('{:.6f}'.format(x), end=' ')
            print()
        elif op[i] in {'cnve', 'cnvm', 'cnvc'}:
            for q1 in grad_param[i]:
                for q2 in q1:
                    for q3 in q2:
                        for q4 in q3:
                            print('{:.6f}'.format(q4), end=' ')
            print()


if __name__ == "__main__":
    main()
