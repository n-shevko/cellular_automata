import os
import torch
import gc

from torchvision import transforms


FRAMES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frames')


def generate_run_id():
    items = os.listdir(FRAMES_FOLDER)
    folders = [int(item) for item in items if os.path.isdir(os.path.join(FRAMES_FOLDER, item))]
    if folders:
        return str(max(folders) + 1)
    else:
        return '1'


# def save_frame(name, image):
#     image_pil = transforms.ToPILImage()(image)
#     image_pil.save(f'/home/nikos/{name}.png', 'PNG')


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