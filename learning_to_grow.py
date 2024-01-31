import random

import torch.nn as nn
from torch import optim

from client import Session
from datetime import datetime
from utils import load_image, init_state_grid, get_cuda
from model import Model


def loop(height, width, image, old):
    cuda = get_cuda()
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