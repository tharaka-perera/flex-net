import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


def create_loss_fn(k, noise_var):
    t_perm = 1e-16 * torch.zeros((k, k))
    for i in range(int(k / 2)):
        t_perm[i, i * 2] = 1.
        t_perm[int(k / 2) + i, i * 2 + 1] = 1.
    t_perm_up = t_perm[:int(k / 2), :]
    t_perm_down = t_perm[int(k / 2):, :]

    u = np.arange(1, k + 1)
    q = 2 * (u % 2) + u - 2

    mask = torch.zeros((k, k))
    mask[np.arange(k), q] = 1.

    def loss_fn(out, h_2):
        power_vec, direction = out
        direction = direction.view(-1, int(k / 2))
        power = power_vec.view(-1, k, 1)
        t_vec = (direction @ t_perm_up + (1 - direction) @ t_perm_down).view(-1, k, 1)

        channels = power * t_vec * h_2
        valid_rx_power = torch.sum(channels * mask, 1)
        interference = torch.sum(channels * (1 - mask), 1) + noise_var
        rate = torch.log2(1 + valid_rx_power / interference)
        sum_rate = torch.mean(torch.sum(rate, 1))
        return -sum_rate

    return loss_fn
