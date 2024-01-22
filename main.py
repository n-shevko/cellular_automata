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


def perceive(state_grid, in_channels):
    filters = prepare_filters(in_channels)
    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=3*in_channels,
        padding=1,
        kernel_size=3,
        bias=False,
        stride=1,
        groups=in_channels
    )
    conv.weight.data = filters
    return conv(state_grid)


class Update(nn.Module):
    def __init__(self):
        super(Update, self).__init__()
        self.dense1 = nn.Linear(in_features=48, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=16)
        nn.init.zeros_(self.dense2.weight)

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)
        ds = self.dense2(x)
        return ds


def stochastic_update(state_grid, ds_grid):
    rand_mask = (torch.rand(*state_grid.size(), device=state_grid.device) < 0.5).float()
    ds_grid = ds_grid * rand_mask
    return state_grid + ds_grid


def alive_masking(state_grid):
    alive = (F.max_pool2d(state_grid[3].unsqueeze(0), kernel_size=3, stride=1, padding=1) > 0.1).float()
    state_grid = state_grid * alive
    return state_grid


def main():
    in_channels = 16
    # [batch_size, channels, height, width]
    height = 3 * 4
    width = 3 * 4
    state_grid = torch.randn(1, in_channels, height, width)
    perception_grid = perceive(state_grid, in_channels)
    perception_grid = perception_grid[0].permute(1, 2, 0).view(-1, 48)
    update = Update()
    ds_grid = update(perception_grid)
    ds_grid = ds_grid.view(width, height, 16).permute(2, 0, 1)
    state_grid = stochastic_update(state_grid[0], ds_grid)
    state_grid = alive_masking(state_grid)


main()