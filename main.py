import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms

from datetime import datetime, timedelta
from PIL import Image

from utils import *

cuda = torch.cuda.is_available()
if cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)
    default_device = torch.device("cuda:0")
    print('GPU')
else:
    torch.set_num_threads((torch.get_num_threads() * 2) - 1)
    print('CPU')


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


def load_image(height, width):
    image = Image.open("lizard.png").convert('RGBA')
    resized_image = image.resize((width, height))
    transform = transforms.ToTensor()
    return transform(resized_image)


def init_state_grid(in_channels, height, width):
    rgb_channels = torch.zeros(3, height, width)
    non_rgb_channel = torch.zeros(height, width)
    seed_y = int(height / 2)
    seed_x = int(width / 2)
    non_rgb_channel[seed_y][seed_x] = 1
    non_rgb_channel = non_rgb_channel.unsqueeze(0)
    non_rgb_channels = non_rgb_channel.repeat(in_channels - 3, 1, 1)
    return torch.cat((rgb_channels, non_rgb_channels), dim=0).unsqueeze(0)

#torch.set_printoptions(threshold=10_000)
torch.set_printoptions(edgeitems=50, linewidth=200, precision=8, threshold=5000)


def loop(height, width):
    #run_id = generate_run_id()
    #os.makedirs(os.path.join(FRAMES_FOLDER, run_id), exist_ok=True)

    target = load_image(height, width)
    target = target.unsqueeze(0)

    in_channels = 16
    state_grid = init_state_grid(in_channels, height, width)
    model = Model(in_channels, width, height)
    if cuda:
        model = model.to('cuda')
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    frame_id = 0
    start = datetime.now()

    rgba = state_grid[:, 0:4, :, :]
    loss = mse(rgba, target)
    while (datetime.now() - start) < timedelta(minutes=20) and round(loss.item(), 4) != 0:
        optimizer.zero_grad()
        out = state_grid
        init_state_grid_val = state_grid.clone()
        # save_frame('init_state_grid_val', init_state_grid_val[:, 0:4, :, :][0])
        # save_frame('target', target[0])
        #save_frame(run_id, frame_id, out[:, 0:4, :, :][0])
        for _ in range(random.randint(64, 96)):
            before = out.clone()
            out = model(out)
            if before.sum() != 0 and out.sum() == 0:
                out2 = model(before)
                v = 3
            frame_id += 1
            #save_frame(run_id, frame_id, out[:, 0:4, :, :][0])
        rgba = out[:, 0:4, :, :]
        loss = mse(rgba, target)
        print(f'Timedelta {(datetime.now() - start)}, Loss: {loss.item():.4f}')
        if round(loss.item(), 4) == 0.1453:
            x = 4
        loss.backward()
        optimizer.step()
    # torch.save(
    #     model.state_dict(),
    #     os.path.join(
    #         os.path.dirname(os.path.abspath(__file__)),
    #         f'model_{round(loss.item(), 4)}.pth'
    #     )
    # )
    #generate_video(run_id)


#loop(32, 32)

def make_video():
    in_channels = 16
    width = 32
    height = 32
    model = Model(in_channels, width, height)
    model.load_state_dict(torch.load('/home/nikos/model_0.0001.pth', map_location=torch.device('cpu')))
    state_grid = init_state_grid(in_channels, height, width)
    mse = nn.MSELoss()
    target = load_image(height, width)
    target = target.unsqueeze(0)
    out = state_grid.clone()
    run_id = generate_run_id()
    os.makedirs(os.path.join(FRAMES_FOLDER, run_id), exist_ok=True)
    frame_id = 0
    save_frame(run_id, frame_id, out[:, 0:4, :, :][0])
    for _ in range(random.randint(64, 96)):
        out = model(out)
        frame_id += 1
        save_frame(run_id, frame_id, out[:, 0:4, :, :][0])
    rgba = out[:, 0:4, :, :]
    loss = round(mse(rgba, target).item(), 4)
    print(loss)
    generate_video(run_id)
    v = 4

#make_video()

# from client import Session
#
# with Session() as s:
#     x = s.take()
#     b = 3

