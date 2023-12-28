import torch
import torch.nn as nn


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


def main():
    in_channels = 16
    # [batch_size, channels, height, width]
    height = 3 * 4
    width = 3 * 4
    state_grid = torch.randn(1, in_channels, height, width)
    grads = perceive(state_grid, in_channels).detach().numpy()
    print(grads.shape)


main()