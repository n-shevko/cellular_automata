import os
import gc
import tempfile

import torch

from tqdm import tqdm
from torchvision import transforms
from PIL import Image

from client import Session
from model import Model


FRAMES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frames')


def generate_run_id():
    items = os.listdir(FRAMES_FOLDER)
    folders = [int(item) for item in items if os.path.isdir(os.path.join(FRAMES_FOLDER, item))]
    if folders:
        return str(max(folders) + 1)
    else:
        return '1'


def save_frame(run_id, frame_id, image):
    image_pil = transforms.ToPILImage()(image)
    image_pil.save(os.path.join(FRAMES_FOLDER, run_id, f'{frame_id}.png'), 'PNG')


def generate_video(folder, file):
    old = os.getcwd()
    os.chdir(folder)
    out = os.path.join(folder, file + '.mp4')
    os.system(f'ffmpeg -framerate 30 -i %d.png -c:v libx264 -pix_fmt yuv420p {out}')
    os.chdir(old)
    return out


def count_objects():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if len(obj.shape) == 1 and obj.shape[0] in [1024, 32]:
                    c = 3
                print(type(obj), obj.size())
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                if len(obj.data.shape) == 1 and obj.data.size()[0] in [1024, 32]:
                    c = 3
                print(type(obj.data), obj.data.size())
        except:
            pass


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


def save_frame(folder, out, name):
    image_pil = transforms.ToPILImage()(out[:, 0:4, :, :][0])
    image_pil.save(os.path.join(folder, f'{name}.png'), 'PNG')


def create_video(image):
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
        for _ in tqdm(range(40000)):
            out = model(out)
            save_frame(temp_dir, out, frame)
            frame += 1
        video = generate_video(temp_dir, image + '_persist')
        os.system(f'mv {video} last_frames')
    os.system(f'rm -R {temp_dir}')


def get_cuda():
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.set_device(0)
        torch.device("cuda:0")
        print('GPU')
    else:
        torch.set_num_threads((torch.get_num_threads() * 2) - 1)
        print('CPU')


def get_mask(height, width):
    y = torch.arange(0, height)
    x = torch.arange(0, width)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    yy = height - yy - 1
    return xx, yy


def damage(mask, grid, radius, x_center, y_center):
    xx, yy = mask
    distances = (xx - x_center) ** 2 + (yy - y_center) ** 2
    circle_mask = distances <= radius ** 2

    circle_indices = circle_mask.nonzero()
    grid[:, :, circle_indices[:, 0], circle_indices[:, 1]] = 0
    return grid