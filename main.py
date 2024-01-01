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


def convert_perception_grid(perception_grid):
    perception_vectors = {}
    for input_channel_filter_combination in perception_grid[0]:
        for i, row in enumerate(input_channel_filter_combination):
            for j, val in enumerate(row):
                position = (i, j)
                perception_vectors.setdefault(position, []).append(float(val))
    return torch.tensor(list(perception_vectors.values()), dtype=torch.float32)


def update_1(perception_vector, out_features):
    dense_1 = nn.Linear(len(perception_vector), out_features)
    x = dense_1(perception_vector)
    x = F.relu(x)
    dense_2 = nn.Linear(out_features, 16)
    nn.init.constant_(dense_2.weight, 0.0)
    ds = dense_2(x)
    return ds


def update_2(perception_vector, dense_1, dense_2):
    x = dense_1(perception_vector)
    x = F.relu(x)
    ds = dense_2(x)
    return ds


def main():
    in_channels = 16
    # [batch_size, channels, height, width]
    height = 3 * 4
    width = 3 * 4
    state_grid = torch.randn(1, in_channels, height, width)
    perception_grid = perceive(state_grid, in_channels)
    perception_grid = convert_perception_grid(perception_grid)

    out_features = 128
    dense_1 = nn.Linear(perception_grid.shape[1], out_features)
    dense_2 = nn.Linear(out_features, 16)
    nn.init.constant_(dense_2.weight, 0.0)

    ds_grid_1 = []
    ds_grid_2 = []
    for perception_vector in perception_grid:
        ds_grid_1.append(update_1(perception_vector, out_features))
        ds_grid_2.append(update_2(perception_vector, dense_1, dense_2))
    x = 3


main()