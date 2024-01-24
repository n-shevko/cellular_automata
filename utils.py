import os

from torchvision import transforms


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


def generate_video(run_id):
    os.chdir(os.path.join(FRAMES_FOLDER, run_id))
    out = os.path.join(FRAMES_FOLDER, run_id + '.mp4')
    os.system(f'ffmpeg -framerate 30 -i %d.png -c:v libx264 -pix_fmt yuv420p {out}')

