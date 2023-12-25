import torch
import torch.nn as nn


# https://distill.pub/2020/growing-ca/


def prepare_filters(in_channels):
    sobel_x = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
    sobel_y = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]
    identity = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    filters = [sobel_x, sobel_y, identity]
    # [out_channels, in_channels, kernel_height, kernel_width]
    res = [[filter for _ in range(in_channels)] for filter in filters]
    return torch.tensor(res, dtype=torch.float32)


def perceive(state_grid, in_channels):
    filters = prepare_filters(in_channels)
    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=len(filters),
        kernel_size=3,
        bias=False,
        stride=(3, 3),
    )
    conv.weight.data = filters
    return conv(state_grid)


def main():
    in_channels = 16
    # [batch_size, channels, height, width]
    height = 3 * 4
    width = 3 * 4
    state_grid = torch.randn(1, in_channels, height, width)
    print(state_grid.numpy())
    print('*' * 100)
    grads = perceive(state_grid, in_channels).detach().numpy()
    print(grads.shape)
    print(grads)


main()