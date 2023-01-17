import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

import model as mod
from loss import create_loss_fn
from utils.data import flex_graph, \
    gen_rectangular_channel_matrix

torch.manual_seed(108)


def generate_data(n, k, batch_size):
    h_batch = gen_rectangular_channel_matrix(k, k, n, seed=11)
    datalist = flex_graph(h_batch)
    return DataLoader(datalist, batch_size=batch_size, shuffle=True)  # make sure to turn on shuffle after dev


def train(model, optimizer, loss_fn, dataset, k, path):
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    for epoch in range(120):
        running_loss = 0.0
        for i, data in enumerate(dataset, 0):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, data.y)
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0
        writer.flush()
    torch.save(model.state_dict(), path)


def eval_model(path, data, k=32, aggr='add', layers=3):
    n = 5000
    model = getattr(mod, f'FlexNet{layers}')(aggr)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)
    model.eval()

    new_data = iter(DataLoader(data, batch_size=n, shuffle=False)).next()
    outs = model(new_data)
    p, t = outs
    t = torch.where(t >= 0.5, 1., 0.)
    p = torch.where(p >= 0.5, 1., 0.)
    rate = create_loss_fn(k, 1.)
    sum_r = rate((p, t), new_data.y)
    print(sum_r.item())
    return -sum_r


def create_and_train_model(n, k, batch_size, noise_var, path, lr=0.01, data=None, aggr='add', layers=3):
    dataset = data if (data is not None) else generate_data(n, k, batch_size)
    model = getattr(mod, f'FlexNet{layers}')(aggr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = create_loss_fn(k, 1.)
    train(model, optimizer, loss_fn, dataset, k, path)
