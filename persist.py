import random
import copy
import os

import tempfile
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from client import Session
from datetime import datetime
from utils import get_cuda, load_image, init_state_grid, create_video, save_frame, generate_video
from model import Model


def experiment_2(height, width, image):
    pool_size = 1024
    batch_size = 32
    target = load_image(height, width, image)
    target = target.expand(batch_size, -1, -1, -1)

    in_channels = 16
    pool = init_state_grid(in_channels, height, width).repeat(pool_size, 1, 1, 1)
    seed = pool[0].clone()

    model = Model(in_channels, width, height)

    if get_cuda():
        pool = pool.to('cuda')
        target = target.to('cuda')
        model = model.to('cuda')

    mse = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters())

    start = datetime.now()
    stop = False
    while not stop:
        idxs = torch.randperm(len(pool))[:batch_size]
        batch = pool[idxs]

        optimizer.zero_grad()
        steps = random.randint(64, 96)
        for _ in range(steps):
            batch = model(batch)

        rgba = batch[:, 0:4, :, :]
        loss = mse(rgba, target).mean(dim=(1, 2, 3))
        max_loss_idx = torch.argmax(loss)
        loss = loss.mean()
        loss_val = loss.item()
        print(f'Timedelta {(datetime.now() - start)}, Loss: {loss_val:.4f}, Target: {image}')
        loss.backward()
        optimizer.step()
        batch[max_loss_idx] = seed
        pool[idxs] = batch.detach()

        stop = round(loss_val, 3) <= 0.001
        if stop:
            with Session() as s:
                s.update(f'exp2_{image}', {
                    'model': copy.deepcopy(model).to('cpu').state_dict(),
                    'loss': loss.item(),
                    'delta': datetime.now() - start,
                })


#experiment_2(32, 32, 'lizard')

#create_video('lizard')

def damage(out, radius):
    height, width = out.shape


def damage_for_immortal(image):
    with Session() as s:
        item = s.take(f'exp2_{image}')

    width = 64
    height = 64
    in_channels = 16
    model = Model(in_channels, width, height)
    model.load_state_dict(item['model'])

    state_grid = init_state_grid(in_channels, height, width)
    temp_dir = tempfile.mkdtemp()
    with torch.no_grad():
        out = state_grid.clone()
        save_frame(temp_dir, out, 0)
        frame = 1
        for _ in tqdm(range(100)):
            out = model(out)
            save_frame(temp_dir, out, frame)
            frame += 1

        for _ in tqdm(range(10000)):
            pass
        video = generate_video(temp_dir, image + '_persist')
        os.system(f'mv {video} last_frames')
    os.system(f'rm -R {temp_dir}')


import torch

torch.set_printoptions(threshold=5000, linewidth=200)

# Установка размеров
height, width = 200, 200
grid = torch.zeros((height, width))

radius = 100
x_center, y_center = 100, 100

# Создание сетки координат
y = torch.arange(0, height)
x = torch.arange(0, width)
xx, yy = torch.meshgrid(x, y, indexing='ij')

# Инверсия координат Y для соответствия координатам плоскости
yy = height - yy - 1

# Вычисление расстояний до центра
distances = (xx - x_center) ** 2 + (yy - y_center) ** 2

# Маска для круга
circle_mask = distances <= radius ** 2

# Получение индексов для круга
circle_indices = circle_mask.nonzero()

# Установка значений в grid
grid[circle_indices[:, 0], circle_indices[:, 1]] = 1
