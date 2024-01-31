import random
import copy

import torch
import torch.nn as nn
from torch import optim

from client import Session
from datetime import datetime
from utils import get_cuda, load_image, init_state_grid
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
