import torch
import torch.nn as nn
import torch.nn.functional as F

# https://distill.pub/2020/growing-ca/


def prepare_filters(in_channels):
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32)
    sobel_y = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=torch.float32)
    identity = torch.tensor([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=torch.float32)
    filters = [sobel_x, sobel_y, identity] * in_channels
    filters = torch.stack(filters).unsqueeze(1)
    # [out_channels, in_channels, kernel_height, kernel_width]
    return filters


def stochastic_update(state_grid, ds_grid):
    rand_mask = (torch.rand(*state_grid.size()) < 0.5).float()
    ds_grid = ds_grid * rand_mask
    return state_grid + ds_grid


def alive_masking(state_grid):
    alive = (F.max_pool2d(state_grid[3].unsqueeze(0), kernel_size=3, stride=1, padding=1) > 0.1).float()
    state_grid = state_grid * alive
    return state_grid


class Model(nn.Module):
    def __init__(self, in_channels, width, height):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=3 * in_channels,
            padding=1,
            kernel_size=3,
            bias=False,
            stride=1,
            groups=in_channels
        )
        self.width = width
        self.height = height
        self.conv.weight.data = prepare_filters(in_channels)
        self.dense1 = nn.Linear(in_features=48, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=16)
        nn.init.zeros_(self.dense2.weight)

    def forward(self, state_grid):
        # Feature Extraction Layer
        perception_grid = self.conv(state_grid)
        perception_grid = perception_grid[0].permute(1, 2, 0).view(-1, 48)

        # Fully Connected Layers
        x = self.dense1(perception_grid)
        x = F.relu(x)
        ds_grid = self.dense2(x)
        ds_grid = ds_grid.view(self.width, self.height, 16).permute(2, 0, 1)

        state_grid = stochastic_update(state_grid[0], ds_grid)
        state_grid = alive_masking(state_grid)
        return state_grid.unsqueeze(0)


def main2():
    in_channels = 16
    height = 3 * 4
    width = 3 * 4
    state_grid = torch.randn(1, in_channels, height, width)
    model = Model(in_channels, width, height)
    out = model(state_grid)
    v = 2


main2()