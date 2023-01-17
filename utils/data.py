import itertools

import networkx as nx
import numpy as np
import torch.distributions
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected, from_networkx

from utils.poisson_disk_gen import PoissonDiskGenerator


def generate_flexible_duplex_fading_channel_variances(users, n, seed=None):
    rng = np.random.default_rng(seed=seed)
    pdg = PoissonDiskGenerator(seed=seed)
    variances = np.empty((n, users, users))
    for i in range(n):
        # limit minimum distance to 100m, area is 4km x 4km
        locations = pdg.generate(users, bead_size=0.1, size=(4, 4))
        # generate separate meshes from x and y coordinates
        x_coords = np.meshgrid(locations[:, 0], locations[:, 0])
        y_coords = np.meshgrid(locations[:, 1], locations[:, 1])
        # calculate the distances between all the combinations of coords
        distances = np.sqrt((x_coords[0] - x_coords[1]) ** 2 + (y_coords[0] - y_coords[1]) ** 2)
        # assign an out of bound number (> 4km) for distance with itself(to avoid numerical errors)
        distances[distances == 0.] = 6.
        #  Free-space path loss(for distance in km and 5GHz freq) + 9.5 dB log-normal shadowing
        path_loss_and_shadowing = 20 * np.log10(distances) + 20 * np.log10(5) + 92.45 + rng.normal(0,
                                                                                                   9.5, (
                                                                                                       users,
                                                                                                       users))
        variances[i] = 10 ** (-path_loss_and_shadowing / 10)
    return variances


def gen_rectangular_channel_matrix(users, antennas, n, seed=None):
    rng = np.random.default_rng(seed=seed)
    variances = 10 ** (-generate_flexible_duplex_fading_channel_variances(users, n, seed) / 10)
    # generate a complex Gaussian
    ch = rng.normal(loc=0,
                    scale=np.sqrt(variances / 2),
                    size=(n, users, antennas)) + 1j * rng.normal(loc=0,
                                                                 scale=np.sqrt(
                                                                     variances / 2),
                                                                 size=(n, users,
                                                                       antennas))
    ch = np.abs(ch)
    # make diagonals zero again (this was intentionally made non-zero to avoid overflows)
    ch[:, np.arange(users), np.arange(users)] = 0.
    return ch.astype(np.float32)


def get_antenna_combinations(count):
    antennas = np.arange(count)
    perms = [list(x) for x in itertools.permutations(antennas, count)]
    return [antennas[perms[x]] for x in np.arange(len(perms))]


def get_rate_batched(h_2, power_vec, k, noise_var):
    power = power_vec.reshape((-1, 1, k))
    rx_power = h_2 * power
    mask = np.eye(k)
    valid_rx_power = np.sum(rx_power * mask, 1)
    interference = np.sum(rx_power * (1 - mask), 2) + noise_var
    return np.log2(1 + (valid_rx_power / interference))


def flex_graph(data_list, initial_power=1., noise_var=1.):
    n = data_list.shape[0]
    k = data_list.shape[1]
    h_2 = data_list ** 2
    h_2_t = h_2.transpose((0, 2, 1))
    obj_list = np.empty(n, dtype=Data)

    complete_graph = nx.complete_graph(range(k))
    complete_pyg = from_networkx(complete_graph)
    complete_pyg.x = torch.ones((complete_pyg.num_nodes, 1))

    u = np.arange(1, k + 1)
    q = 2 * (u % 2) + u - 2

    sig_mask = np.zeros((k, k), dtype=np.bool)
    sig_mask[np.arange(k), q] = True

    node_values = h_2[:, np.arange(k), q]

    triu_mask = np.zeros((k, k), dtype=np.bool)
    triu_mask[np.triu_indices_from(triu_mask, k=1)] = True
    triu_mask = ~sig_mask & triu_mask

    tril_mask = np.zeros((k, k), dtype=np.bool)
    tril_mask[np.tril_indices_from(tril_mask, k=-1)] = True
    tril_mask = ~sig_mask & tril_mask

    edge_attr_up = h_2[:, triu_mask]
    edge_attr_low = h_2[:, tril_mask]
    edge_attr_up2 = h_2_t[:, triu_mask]
    edge_attr_low2 = h_2_t[:, tril_mask]

    upper_tri_indices = triu_mask.nonzero()
    lower_tri_indices = tril_mask.nonzero()

    for i in range(n):
        data = Data()
        data.x = torch.from_numpy(np.expand_dims(node_values[i], -1))
        data.edge_index = torch.cat(
            (torch.from_numpy(np.asarray(upper_tri_indices)),
             torch.from_numpy(np.asarray(lower_tri_indices)))
            , dim=1).T.flip(-1).T
        data.edge_attr = torch.cat((
            torch.cat((torch.from_numpy(edge_attr_up[i]), torch.from_numpy(edge_attr_low[i]))).unsqueeze(-1),
            torch.cat((torch.from_numpy(edge_attr_up2[i]), torch.from_numpy(edge_attr_low2[i]))).unsqueeze(-1)
        ), dim=1)
        data.y = torch.from_numpy(h_2[i].reshape(1, k, k))
        data.dir_edge_index = torch.cat((torch.arange(0, k, 2, dtype=torch.long).unsqueeze(0),
                                         torch.arange(1, k, 2, dtype=torch.long).unsqueeze(0)))
        data.prop_edge_index = torch.cat(
            (torch.cat((torch.arange(0, k, 2, dtype=torch.long), torch.arange(1, k, 2, dtype=torch.long))).unsqueeze(0),
             torch.cat((torch.arange(1, k, 2, dtype=torch.long), torch.arange(0, k, 2, dtype=torch.long))).unsqueeze(0))
        )
        obj_list[i] = data

    return obj_list