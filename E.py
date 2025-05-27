import numpy as np


def read_matrix(n):
    return np.array([list(map(float, input().split())) for _ in range(n)])


def read_vector(n):
    return np.array(list(map(float, input().split())))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    N = int(input())
    W_f = read_matrix(N)
    U_f = read_matrix(N)
    b_f = read_vector(N)
    W_i = read_matrix(N)
    U_i = read_matrix(N)
    b_i = read_vector(N)
    W_o = read_matrix(N)
    U_o = read_matrix(N)
    b_o = read_vector(N)
    W_c = read_matrix(N)
    U_c = read_matrix(N)
    b_c = read_vector(N)
    M = int(input())
    h_prev = read_vector(N)
    c_prev = read_vector(N)
    xs = [read_vector(N) for _ in range(M)]
    dh_M = read_vector(N)
    dc_M = read_vector(N)
    d_o = [read_vector(N) for _ in range(M)][::-1]
    h_list = [h_prev.copy()]
    c_list = [c_prev.copy()]
    o_list = []
    saved = []
    for t in range(M):
        x_t = xs[t]
        h_prev = h_list[-1]
        c_prev = c_list[-1]
        f_t = sigmoid(W_f @ x_t + U_f @ h_prev + b_f)
        i_t = sigmoid(W_i @ x_t + U_i @ h_prev + b_i)
        o_t = sigmoid(W_o @ x_t + U_o @ h_prev + b_o)
        c_hat_t = np.tanh(W_c @ x_t + U_c @ h_prev + b_c)
        c_t = f_t * c_prev + i_t * c_hat_t
        h_t = o_t * c_t
        h_list.append(h_t)
        c_list.append(c_t)
        o_list.append(o_t)
        saved.append((f_t, i_t, o_t, c_hat_t, x_t, h_prev.copy(), c_prev.copy()))

    for o in o_list:
        print(' '.join(map("{:.20f}".format, o)))
    print(' '.join(map("{:.20f}".format, h_list[-1])))
    print(' '.join(map("{:.20f}".format, c_list[-1])))

    dh_next = dh_M.copy()
    dc_next = dc_M.copy()
    dx_grads = []
    dW_f = np.zeros_like(W_f)
    dU_f = np.zeros_like(U_f)
    db_f = np.zeros_like(b_f)
    dW_i = np.zeros_like(W_i)
    dU_i = np.zeros_like(U_i)
    db_i = np.zeros_like(b_i)
    dW_o = np.zeros_like(W_o)
    dU_o = np.zeros_like(U_o)
    db_o = np.zeros_like(b_o)
    dW_c = np.zeros_like(W_c)
    dU_c = np.zeros_like(U_c)
    db_c = np.zeros_like(b_c)
    for t in reversed(range(M)):
        f_t, i_t, o_t, c_hat_t, x_t, h_prev, c_prev = saved[t]
        c_t = c_list[t + 1]
        do_t = d_o[t] + dh_next * c_t
        d_o_t = do_t * o_t * (1 - o_t)
        dc = dc_next + dh_next * o_t
        dc_hat = dc * i_t * (1 - c_hat_t ** 2)
        di = dc * c_hat_t * i_t * (1 - i_t)
        df = dc * c_prev * f_t * (1 - f_t)
        dW_c += np.outer(dc_hat, x_t)
        dU_c += np.outer(dc_hat, h_prev)
        db_c += dc_hat
        dW_i += np.outer(di, x_t)
        dU_i += np.outer(di, h_prev)
        db_i += di
        dW_f += np.outer(df, x_t)
        dU_f += np.outer(df, h_prev)
        db_f += df
        dW_o += np.outer(d_o_t, x_t)
        dU_o += np.outer(d_o_t, h_prev)
        db_o += d_o_t
        dx = (W_c.T @ dc_hat) + (W_i.T @ di) + (W_f.T @ df) + (W_o.T @ d_o_t)
        dh_prev = (U_c.T @ dc_hat) + (U_i.T @ di) + (U_f.T @ df) + (U_o.T @ d_o_t)
        dx_grads.append(dx)
        dc_next = dc * f_t
        dh_next = dh_prev

    for dx in dx_grads:
        print(' '.join(map("{:.20f}".format, dx)))

    print(' '.join(map("{:.20f}".format, dh_next)))
    print(' '.join(map("{:.20f}".format, dc_next)))

    for grad in [dW_f, dU_f, db_f, dW_i, dU_i, db_i, dW_o, dU_o, db_o, dW_c, dU_c, db_c]:
        if grad.ndim == 2:
            for row in grad:
                print(' '.join(map("{:.20f}".format, row)))
        else:
            print(' '.join(map("{:.20f}".format, grad)))


if __name__ == "__main__":
    main()
