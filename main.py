import torch
from torch_geometric.loader import DataLoader

from loss import create_loss_fn
from model import FlexNet
from utils.data import flex_graph, \
    gen_rectangular_channel_matrix


def generate_data(n, k, batch_size):
    h_batch = gen_rectangular_channel_matrix(k, k, n, seed=11)
    datalist = flex_graph(h_batch)
    return DataLoader(datalist, batch_size=batch_size, shuffle=True)


def train(model, optimizer, loss_fn, dataset, k, path):
    model.train()
    for epoch in range(120):
        running_loss = 0.0
        for i, data in enumerate(dataset, 0):
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, data.y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0
    torch.save(model.state_dict(), path)


def eval_model(path, k, noise_var):
    n = 10000
    model = FlexNet()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    data_t = gen_rectangular_channel_matrix(k, k, n, seed=899)

    new_data = flex_graph(data_t)
    new_data = next(iter(DataLoader(new_data, batch_size=n, shuffle=False)))

    outs = model(new_data)
    p, t = outs
    t = torch.where(t >= 0.5, 1., 0.)
    p = torch.where(p >= 0.5, 1., 0.)
    rate = create_loss_fn(k, 1.)
    sum_rate = -rate((p, t), new_data.y).item()
    print('Test sum-rate:', sum_rate)
    return sum_rate


def create_and_train_model(n, k, batch_size, noise_var, path, lr=0.004, data=None, aggr='add'):
    dataset = data if (data is not None) else generate_data(n, k, batch_size)
    model = FlexNet(aggr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = create_loss_fn(k, 1.)
    train(model, optimizer, loss_fn, dataset, k, path)


if __name__ == '__main__':
    path_ = 'flexible____.pth'
    # create_and_train_model(n=10000, k=20, batch_size=64, noise_var=1., path=path_)
    eval_model(path=path_, k=20, noise_var=1.)

