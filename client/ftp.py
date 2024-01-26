import io
import sys

from datetime import timedelta, datetime
from os.path import join

import paramiko

from client.utils import CONFIG, PKEY


def get_sftp():
    transport = paramiko.Transport((CONFIG['host'], CONFIG['ftp']['port']))
    transport.connect(
        username=CONFIG['ftp']['username'],
        pkey=PKEY
    )
    return paramiko.SFTPClient.from_transport(transport), transport


def print_speed(verb, obj, delta):
    if delta > timedelta(seconds=5):
        mb_size = sys.getsizeof(obj) / (1024 * 1024)
        mb_per_second = mb_size / delta.total_seconds()
        if mb_per_second < 3:
            print('Slow %s speed on fs:\nTime delta: %s\nValue size: %s mb\nSpeed: %s mb / sec' % (
            verb, delta, mb_size, mb_per_second))


class Ftp(object):
    ftp = None

    def __enter__(self):
        return self

    def read(self, path: str, default_value=None):
        start = datetime.now()
        try:
            if self.ftp is None:
                self.ftp = get_sftp()

            channel, transport = self.ftp
            val = io.BytesIO()
            channel.getfo(path, val)
            result = val.getvalue()
            delta = datetime.now() - start
        except FileNotFoundError:
            return default_value
        except Exception as e:
            raise e
        print_speed('read', result, delta)
        return result

    def write(self, obj: bytes, base_folder, rest):
        start = datetime.now()
        path = join(base_folder, rest)
        if self.ftp is None:
            self.ftp = get_sftp()
        channel, transport = self.ftp

        folders = []
        for folder in rest.split('/')[:-1]:
            if not folder:
                continue

            try:
                folders.append(folder)
                folder_to_create = join(*([base_folder] + folders))
                channel.mkdir(folder_to_create)
            except Exception as _:
                pass
        channel.putfo(io.BytesIO(obj), path)

        delta = datetime.now() - start
        print_speed('write', obj, delta)

    def delete_fs(self, path_or_paths):
        if not path_or_paths:
            return

        if isinstance(path_or_paths, str):
            path_or_paths = [path_or_paths]

        if self.ftp is None:
            self.ftp = get_sftp()
        channel, transport = self.ftp
        for path in path_or_paths:
            try:
                channel.remove(path)
            except Exception:
                pass

    def __exit__(self, *exc):
        if self.ftp is not None:
            channel, transport = self.ftp
            channel.close()
            transport.close()
        return False
