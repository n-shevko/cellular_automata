import torch
import torch.nn as nn

# https://distill.pub/2020/growing-ca/

def perceive(state_grid):
    sobel_x = torch.tensor(
        [[-1, 0, +1],
         [-2, 0, +2],
         [-1, 0, +1]]
    )
    # out_channels = 1 assumed
    sobel_y = sobel_x.transpose(0, 1)


    in_channels = 3
    out_channels = 64
    kernel_size = 3
    stride = 1
    padding = 1

    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)


    input_data = torch.randn(1, 3, 32, 32)
    output_data = conv_layer(input_data)


def main():
    state_grid = torch.randn(10, 10, 16)
    print(tensor_3d)
    v = 3

main()