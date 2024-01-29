import random

import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms

from client import Session
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
    only_alpha = state_grid[:, 3, :, :]
    alive = (F.max_pool2d(only_alpha, kernel_size=3, stride=1, padding=1) > 0.1).float()
    alive = alive.unsqueeze(1).expand(-1, state_grid.shape[1], -1, -1)
    state_grid = state_grid * alive
    return state_grid


def alive_masking_old(state_grid):
    alive = (F.max_pool2d(state_grid[3].unsqueeze(0), kernel_size=3, stride=1, padding=1) > 0.1).float()
    state_grid = state_grid * alive
    return state_grid


class Model(nn.Module):
    def __init__(self, in_channels, width, height, old):
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
        self.old = old

    def forward_old(self, state_grid):
        # Feature Extraction Layer
        perception_grid = self.conv(state_grid)
        perception_grid = perception_grid[0].permute(1, 2, 0).view(-1, 48)

        # Fully Connected Layers
        x = self.dense1(perception_grid)
        x = F.relu(x)
        ds_grid = self.dense2(x)
        ds_grid = ds_grid.view(self.width, self.height, 16).permute(2, 0, 1)

        state_grid = stochastic_update(state_grid[0], ds_grid)
        state_grid = alive_masking_old(state_grid)
        return state_grid.unsqueeze(0)

    def forward(self, state_grid):
        if self.old:
            return self.forward_old(state_grid)

        # Feature Extraction Layer
        perception_grid = self.conv(state_grid)
        perception_grid = perception_grid.permute(0, 2, 3, 1)

        # Fully Connected Layers
        x = self.dense1(perception_grid)
        x = F.relu(x)
        ds_grid = self.dense2(x)
        ds_grid = ds_grid.permute(0, 3, 1, 2)

        state_grid = stochastic_update(state_grid, ds_grid)
        state_grid = alive_masking(state_grid)
        return state_grid


def load_image(height, width, image_name):
    image = Image.open(f"images/{image_name}.png").convert('RGBA')
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


def loop(height, width, image, old):
    target = load_image(height, width, image)
    target = target.unsqueeze(0)
    in_channels = 16
    state_grid = init_state_grid(in_channels, height, width)
    if cuda:
        state_grid = state_grid.to('cuda')
        target = target.to('cuda')

    store = None
    while True:
        failed = False
        loss_tracing = []
        model = Model(in_channels, width, height, old)
        if cuda:
            model = model.to('cuda')
        mse = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())
        start = datetime.now()

        last_loss = 100
        last_steps = None
        while round(last_loss, 3) > 0.001 and not failed:
            optimizer.zero_grad()
            out = state_grid.clone()
            last_steps = random.randint(64, 96)
            for _ in range(last_steps):
                out = model(out)
                if out.sum() == 0:
                    failed = True
            rgba = out[:, 0:4, :, :]
            loss = mse(rgba, target)
            last_loss = loss.item()
            print(f'Timedelta {(datetime.now() - start)}, Loss: {last_loss:.4f}, Target: {image}')
            loss_tracing.append(last_loss)
            loss.backward()
            optimizer.step()

        if failed:
            store = {
                'failed': True,
                'loss_tracing': loss_tracing
            }
            print('Failed')
            if len(loss_tracing) < 30:
                continue
            else:
                break
        else:
            print('Done')
            store = {
                'state_dict': model.to('cpu').state_dict(),
                'last_loss': last_loss,
                'last_steps': last_steps,
                'width': width,
                'height': height
            }
            break

    with Session() as s:
        s.update(f'exp1_1_{image}', store)


def save_frame(folder, out, name):
    image_pil = transforms.ToPILImage()(out[:, 0:4, :, :][0])
    image_pil.save(os.path.join(folder, f'{name}.png'), 'PNG')


def create_video_and_last_frame(image):
    with Session() as s:
        item = s.take(f'exp1_{image}')

    model = Model(16, item['width'], item['height'])
    model.load_state_dict(item['state_dict'])

    in_channels = 16
    width = item['width']
    height = item['height']

    state_grid = init_state_grid(in_channels, height, width)
    # out = state_grid.clone()
    # with torch.no_grad():
    #     for _ in range(2000):
    #         out = model(out)
    #     save_frame('last_frames', out, image)

    target = load_image(height, width, image)
    target = target.unsqueeze(0)

    temp_dir = tempfile.mkdtemp()
    save_frame(temp_dir, target, 'target')
    with torch.no_grad():
        out = state_grid.clone()
        save_frame(temp_dir, out, 0)
        frame = 1
        from tqdm import tqdm
        mse = nn.MSELoss()



        for step in tqdm(range(item['last_steps'] + 20000)):
            if step + 1 < item['last_steps']:
                rgba = out[:, 0:4, :, :]
                loss = mse(rgba, target).item()
                print(f'Loss {loss}')
            out = model(out)
            save_frame(temp_dir, out, frame)
            frame += 1
        video = generate_video(temp_dir, image)
        os.system(f'mv {video} last_frames')

        rgba = out[:, 0:4, :, :]
        loss = mse(rgba, target).item()
        print(f'Loss {loss}')
        c = 3
    os.system(f'rm -R {temp_dir}')



#make_video()
#make_video()

# for image in ['lizard', 'butterfly', 'violin', 'shamrock', 'eggplant']:
#     loop(64, 64, 'lizard')
# for image in ['violin', 'shamrock', 'eggplant']:
#     create_video_and_last_frame(image)



#loop(64, 64, 'lizard', False)
