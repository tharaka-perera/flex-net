from torch import sigmoid, cat
from torch.nn import Sequential, Linear, ReLU, Module
from torch_geometric.nn import MessagePassing, BatchNorm


class FlexEdge(MessagePassing):
    def __init__(self, in_channels):
        super().__init__()
        hidden_channels = int(in_channels / 2)
        self.mlp1 = Sequential(Linear(in_channels, hidden_channels),
                               ReLU(),
                               BatchNorm(hidden_channels),
                               Linear(hidden_channels, 1))

    def forward(self, x, edge_index, power):
        return self.edge_updater(edge_index, x=x, power=power)

    def edge_update(self, x_i, x_j, power_i, power_j):
        return self.mlp1(cat((x_i, power_i, x_j, power_j), 1))


class FlexConv(MessagePassing):
    def __init__(self, in_channels, aggr='add'):
        super().__init__(aggr=aggr)
        self.edge_weight = Linear(2, in_channels + 1, bias=False)
        self.neighbour_weight = Linear(in_channels, in_channels + 1, bias=False)
        self.self_weight = Linear(in_channels, in_channels + 1, bias=False)
        self.activation = Sequential(BatchNorm(in_channels + 1), ReLU())

    def forward(self, x, edge_index, edge_attr):
        return cat((x[:, 0].view(-1, 1), self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr)), dim=1)

    def message(self, x_i, x_j, edge_index, edge_attr):
        out = self.self_weight(x_i) + self.edge_weight(edge_attr) + self.neighbour_weight(x_j)
        return self.activation(out)


class EdgeInter(MessagePassing):
    def __init__(self, in_channels):
        super().__init__()
        hidden = int(in_channels * 1.5)

        self.neighbour_weight = Linear(in_channels, in_channels + 1, bias=False)
        self.self_weight = Linear(in_channels, in_channels + 1, bias=False)
        self.activation = Sequential(BatchNorm(in_channels + 1), ReLU())

    def forward(self, x, edge_index):
        return cat((x[:, 0].view(-1, 1), self.propagate(x=x, edge_index=edge_index)), dim=1)

    def message(self, x_j, x_i, edge_index):
        return self.activation(self.neighbour_weight(x_j) + self.self_weight(x_i))


# Three layered FlexNet model
class FlexNet(Module):
    def __init__(self, aggr='add'):
        super().__init__()
        self.conv1 = FlexConv(1, aggr)
        self.desired_edge1 = EdgeInter(3)

        self.conv2 = FlexConv(5, aggr)
        self.desired_edge2 = EdgeInter(7)

        self.conv3 = FlexConv(9, aggr)
        self.desired_edge3 = EdgeInter(11)

        self.power_mlp = Sequential(Linear(13, 8), ReLU(), BatchNorm(8), Linear(8, 4), ReLU(), BatchNorm(4),
                                    Linear(4, 1))
        self.edge_conv = FlexEdge(28)
        self.mlp_node = Sequential(Linear(1, 4), ReLU())
        self.mlp_edge = Sequential(Linear(2, 4), ReLU())

    def forward(self, data):
        x0, edge_attr, edge_index, dir_edge_index, prop_edge_index = data.x, data.edge_attr, data.edge_index, data.dir_edge_index, data.prop_edge_index

        out = self.conv1(x0, edge_index, edge_attr)
        out = self.desired_edge1(out, prop_edge_index)

        out = self.conv2(out, edge_index, edge_attr)
        out = self.desired_edge2(out, prop_edge_index)

        out = self.conv3(out, edge_index, edge_attr)
        out = self.desired_edge3(out, prop_edge_index)

        power = sigmoid(1.5 * self.power_mlp(out))

        direction = self.edge_conv(out, dir_edge_index, power)
        return power, sigmoid(1.5 * direction)


class FlexNet1(Module):
    def __init__(self, aggr='add'):
        super().__init__()
        self.conv1 = FlexConv(1, aggr)
        self.desired_edge1 = EdgeInter(3)

        self.power_mlp = Sequential(Linear(5, 4), ReLU(), BatchNorm(4), Linear(4, 1))
        self.edge_conv = FlexEdge(12)

    def forward(self, data):
        x0, edge_attr, edge_index, dir_edge_index, prop_edge_index = data.x, data.edge_attr, data.edge_index, data.dir_edge_index, data.prop_edge_index

        out = self.conv1(x0, edge_index, edge_attr)
        out = self.desired_edge1(out, prop_edge_index)

        power = sigmoid(1.5 * self.power_mlp(out))

        direction = self.edge_conv(out, dir_edge_index, power)
        return power, sigmoid(1.5 * direction)


class FlexNet2(Module):
    def __init__(self, aggr='add'):
        super().__init__()
        self.conv1 = FlexConv(1, aggr)
        self.desired_edge1 = EdgeInter(3)

        self.conv2 = FlexConv(5, aggr)
        self.desired_edge2 = EdgeInter(7)

        self.power_mlp = Sequential(Linear(9, 4), ReLU(), BatchNorm(4), Linear(4, 1))
        self.edge_conv = FlexEdge(20)

    def forward(self, data):
        x0, edge_attr, edge_index, dir_edge_index, prop_edge_index = data.x, data.edge_attr, data.edge_index, data.dir_edge_index, data.prop_edge_index

        out = self.conv1(x0, edge_index, edge_attr)
        out = self.desired_edge1(out, prop_edge_index)

        out = self.conv2(out, edge_index, edge_attr)
        out = self.desired_edge2(out, prop_edge_index)

        power = sigmoid(1.5 * self.power_mlp(out))

        direction = self.edge_conv(out, dir_edge_index, power)
        return power, sigmoid(1.5 * direction)


class FlexNet3(FlexNet):
    def __init__(self, aggr='add'):
        super().__init__(aggr=aggr)


class FlexNet4(Module):
    def __init__(self, aggr='add'):
        super().__init__()
        self.conv1 = FlexConv(1, aggr)
        self.desired_edge1 = EdgeInter(3)

        self.conv2 = FlexConv(5, aggr)
        self.desired_edge2 = EdgeInter(7)

        self.conv3 = FlexConv(9, aggr)
        self.desired_edge3 = EdgeInter(11)

        self.conv4 = FlexConv(13, aggr)
        self.desired_edge4 = EdgeInter(15)

        self.power_mlp = Sequential(Linear(17, 10), ReLU(), BatchNorm(10), Linear(10, 4), ReLU(), BatchNorm(4),
                                    Linear(4, 1))
        self.edge_conv = FlexEdge(36)

    def forward(self, data):
        x0, edge_attr, edge_index, dir_edge_index, prop_edge_index = data.x, data.edge_attr, data.edge_index, data.dir_edge_index, data.prop_edge_index

        out = self.conv1(x0, edge_index, edge_attr)
        out = self.desired_edge1(out, prop_edge_index)

        out = self.conv2(out, edge_index, edge_attr)
        out = self.desired_edge2(out, prop_edge_index)

        out = self.conv3(out, edge_index, edge_attr)
        out = self.desired_edge3(out, prop_edge_index)

        out = self.conv4(out, edge_index, edge_attr)
        out = self.desired_edge4(out, prop_edge_index)

        power = sigmoid(1.5 * self.power_mlp(out))

        direction = self.edge_conv(out, dir_edge_index, power)
        return power, sigmoid(1.5 * direction)


class FlexNet5(Module):
    def __init__(self, aggr='add'):
        super().__init__()
        self.conv1 = FlexConv(1, aggr)
        self.desired_edge1 = EdgeInter(3)

        self.conv2 = FlexConv(5, aggr)
        self.desired_edge2 = EdgeInter(7)

        self.conv3 = FlexConv(9, aggr)
        self.desired_edge3 = EdgeInter(11)

        self.conv4 = FlexConv(13, aggr)
        self.desired_edge4 = EdgeInter(15)

        self.conv5 = FlexConv(17, aggr)
        self.desired_edge5 = EdgeInter(19)

        self.power_mlp = Sequential(Linear(21, 10), ReLU(), BatchNorm(10), Linear(10, 4), ReLU(), BatchNorm(4),
                                    Linear(4, 1))
        self.edge_conv = FlexEdge(44)

    def forward(self, data):
        x0, edge_attr, edge_index, dir_edge_index, prop_edge_index = data.x, data.edge_attr, data.edge_index, data.dir_edge_index, data.prop_edge_index

        out = self.conv1(x0, edge_index, edge_attr)
        out = self.desired_edge1(out, prop_edge_index)

        out = self.conv2(out, edge_index, edge_attr)
        out = self.desired_edge2(out, prop_edge_index)

        out = self.conv3(out, edge_index, edge_attr)
        out = self.desired_edge3(out, prop_edge_index)

        out = self.conv4(out, edge_index, edge_attr)
        out = self.desired_edge4(out, prop_edge_index)

        out = self.conv5(out, edge_index, edge_attr)
        out = self.desired_edge5(out, prop_edge_index)

        power = sigmoid(1.5 * self.power_mlp(out))

        direction = self.edge_conv(out, dir_edge_index, power)
        return power, sigmoid(1.5 * direction)


class FlexEdgeFixed(MessagePassing):
    def __init__(self, in_channels):
        super().__init__()
        hidden_channels = 12
        self.mlp1 = Sequential(Linear(in_channels, hidden_channels),
                               ReLU(),
                               BatchNorm(hidden_channels),
                               Linear(hidden_channels, 1))

    def forward(self, x, edge_index, power):
        return self.edge_updater(edge_index, x=x, power=power)

    def edge_update(self, x_i, x_j, power_i, power_j):
        return self.mlp1(cat((x_i, power_i, x_j, power_j), 1))


class FlexNetSamp(Module):
    def __init__(self, aggr='add'):
        super().__init__()
        self.conv1 = FlexConv(1, aggr)
        self.desired_edge1 = EdgeInter(3)

        self.conv2 = FlexConv(5, aggr)
        self.desired_edge2 = EdgeInter(7)

        self.conv3 = FlexConv(9, aggr)
        self.desired_edge3 = EdgeInter(11)

        self.power_mlp = Sequential(Linear(13, 8), ReLU(), BatchNorm(8), Linear(8, 4), ReLU(), BatchNorm(4),
                                    Linear(4, 1))
        self.edge_conv = FlexEdgeFixed(28)
        self.mlp_node = Sequential(Linear(1, 4), ReLU())
        self.mlp_edge = Sequential(Linear(2, 4), ReLU())

    def forward(self, data):
        x0, edge_attr, edge_index, dir_edge_index, prop_edge_index = data.x, data.edge_attr, data.edge_index, data.dir_edge_index, data.prop_edge_index

        out = self.conv1(x0, edge_index, edge_attr)
        out = self.desired_edge1(out, prop_edge_index)

        out = self.conv2(out, edge_index, edge_attr)
        out = self.desired_edge2(out, prop_edge_index)

        out = self.conv3(out, edge_index, edge_attr)
        out = self.desired_edge3(out, prop_edge_index)

        power = sigmoid(1.5 * self.power_mlp(out))

        direction = self.edge_conv(out, dir_edge_index, power)
        return power, sigmoid(1.5 * direction)