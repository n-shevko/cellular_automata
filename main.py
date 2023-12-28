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
        kernel_size=3,
        bias=False,
        stride=1,
        groups=in_channels
    )
    conv.weight.data = filters
    return conv(state_grid)


def update(perception_grid):
    _, out_channels, height, width = perception_grid.shape
    in_features = out_channels * height * width
    perception_vectors = perception_grid.view(-1, in_features)
    dense = nn.Linear(in_features, 128)
    x = dense(perception_vectors)
    x = F.relu(x)
    dense2 = nn.Linear(128, 16)
    nn.init.constant_(dense2.weight, 0.0)
    ds = dense2(x)
    c = 3

def main():
    in_channels = 16
    # [batch_size, channels, height, width]
    height = 3 * 4
    width = 3 * 4
    state_grid = torch.randn(1, in_channels, height, width)
    perception_grid = perceive(state_grid, in_channels)#.detach().numpy()
    #update(perception_grid)
    # (1, 48, 10, 10)


main()