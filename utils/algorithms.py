import numpy as np


# from original paper
from numba import njit, prange


@njit
def wmmse(p_int, H, Pmax, var_noise):
    K = np.shape(p_int)[0]
    vnew = 0
    b = np.sqrt(p_int).astype(np.float32)  # v_k
    f = np.zeros(K, dtype=np.float32)  # u_k
    w = np.zeros(K, dtype=np.float32)
    for i in range(K):
        f[i] = H[i, i] * b[i] / (np.square(H[i, :]) @ np.square(b) + var_noise)
        w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
        vnew = vnew + np.log2(w[i])

    VV = np.zeros(100)
    for iter in range(100):
        vold = vnew
        for i in range(K):
            btmp = w[i] * f[i] * H[i, i] / sum(w * np.square(f) * np.square(H[:, i]))
            b[i] = min(btmp, np.sqrt(Pmax)) + max(btmp, 0) - btmp

        vnew = 0
        for i in range(K):
            f[i] = H[i, i] * b[i] / ((np.square(H[i, :])) @ (np.square(b)) + var_noise)
            w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
            vnew = vnew + np.log2(w[i])

        VV[iter] = vnew
        if vnew - vold <= 1e-4:
            break
    p_opt = np.square(b)
    return p_opt


def wmmse2(p_int, H, Pmax, var_noise):
    K = np.size(p_int)
    vnew = 0
    b = np.sqrt(p_int)  # v
    f = np.zeros(K)
    w = np.zeros(K)
    P_pi = np.identity(K)
    for i in range(K):
        f[i] = H[i, i] * b[i] / (np.square(H[i, :]) @ np.square(b) + var_noise)  # u
        w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
        vnew = vnew + np.log2(w[i])

    VV = np.zeros(100)
    for iter in range(100):
        H = H @ P_pi
        vold = vnew
        for i in range(K):
            btmp = w[i] * f[i] * H[i, i] / sum(w * np.square(f) * np.square(H[:, i]))
            b[i] = min(btmp, np.sqrt(Pmax)) + max(btmp, 0) - btmp

        vnew = 0
        for i in range(K):
            f[i] = H[i, i] * b[i] / ((np.square(H[i, :])) @ (np.square(b)) + var_noise)
            w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
            vnew = vnew + np.log2(w[i])

        VV[iter] = vnew
        if vnew - vold <= 1e-3:
            break

    p_opt = np.square(b)
    return p_opt


@njit(parallel=True)
def get_sum_rate(h, p, var_noise, k):
    y = 0.0
    for i in prange(k):
        s = var_noise
        for j in prange(k):
            if j != i:
                s = s + h[i, j] ** 2 * p[j]
        y = y + np.log2(1 + (h[i, i] ** 2 * p[i]) / s)
    return y


@njit
def get_directional_sum_rate(h, p, t, var_noise, k):
    p_t = p * t
    g = h ** 2
    channels = p_t.reshape((k, 1)) * g
    u = np.arange(1, k+1)
    q = 2 * (u % 2) + u - 2
    temp_channel = np.empty(k)
    for i, (row, col) in enumerate(zip(np.arange(k), q)):
        temp_channel[i] = channels[row, col]
    signal_channel = temp_channel[q]
    # signal_channel = channels[np.arange(k), q][q]
    interference = channels[:, q].sum(axis=0)[q] - signal_channel
    sinr = signal_channel/(var_noise + interference)
    rates = np.log2(1 + sinr)
    return rates.sum()


def pattern_search(h, p, k):
    num_of_pairs = int(k / 2)
    max_rate = np.empty(num_of_pairs)
    max_t = np.empty((num_of_pairs, k))
    t_pattern = np.zeros((num_of_pairs + 1, num_of_pairs))
    # t_pattern = np.tile(np.array([0, 1]*num_of_pairs).reshape((1, k)), (num_of_pairs + 1, 1))
    for i in range(0, num_of_pairs):
        t_pattern[i + 1][i] = 1
        # t_pattern[i+1, 2*i] = 1
        # t_pattern[i+1, 2*i+1] = 0
    for j in range(num_of_pairs):
        t_pat = t_pattern[j].copy()
        k_ = j
        while True:
            sum_rates = np.empty(num_of_pairs)
            for i in range(num_of_pairs):
                t = t_pat.copy()
                if i != 0:
                    t[2 * (i - 1)] = 1
                    t[2 * (i - 1) + 1] = 0
                # t[2*i] = 1
                # t[2*i + 1] = 0
                t_sym = np.roll(np.concatenate([1 - t, np.zeros(num_of_pairs)]), 1) + np.concatenate(
                    [t, np.zeros(num_of_pairs)])
                sum_rates[i] = get_directional_sum_rate(h, p, t_sym, 1., k)
            max_idx = np.argmax(sum_rates)
            if max_idx == k_:
                break
            else:
                t_pat[max_idx] = 1
                t_pat[max_idx - 1 if max_idx % 2 == 1 else max_idx + 1] = 0
                k_ = max_idx
        t_sym = np.roll(np.concatenate([1 - t_pat, np.zeros(num_of_pairs)]), 1) + np.concatenate(
            [t_pat, np.zeros(num_of_pairs)])
        max_rate[j] = get_directional_sum_rate(h, p, t_sym, 1., k)
        max_t[j] = t_sym
    max_rate_idx = np.argmax(max_rate)
    return max_rate[max_rate_idx], max_t[max_rate_idx]


def build_t_vec(t):
    return np.vstack([t, 1 - t]).flatten('F')


def pattern_search2(h, p, k):
    num_of_pairs = int(k / 2)
    max_rate = np.empty(num_of_pairs + 1)
    max_t = np.empty((num_of_pairs + 1, k))
    t_pattern = np.zeros((num_of_pairs + 1, num_of_pairs))
    for i in range(0, num_of_pairs):
        t_pattern[i + 1][i] = 1
    for j, pat in enumerate(t_pattern):
        max_idx = j
        t_pat = pat.copy()
        while True:
            sum_rates = np.empty(num_of_pairs + 1)
            sum_rates[0] = get_directional_sum_rate(h, p, build_t_vec(t_pat), 1., k)
            for i in range(num_of_pairs):
                t = t_pat.copy()
                t[i] = 1
                sum_rates[i + 1] = get_directional_sum_rate(h, p, build_t_vec(t), 1., k)
            index = np.argmax(sum_rates)
            if index == 0 or index == max_idx:
                break
            else:
                t_pat[index - 1] = 1
                max_idx = index
        max_rate[j] = get_directional_sum_rate(h, p, build_t_vec(t_pat), 1., k)
        max_t[j] = build_t_vec(t_pat)
    max_rate_idx = np.argmax(max_rate)
    return max_rate[max_rate_idx], max_t[max_rate_idx]
