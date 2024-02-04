import sys

from os.path import join
from collections import OrderedDict

from client.db import Db
from client.ftp import Ftp, get_sftp
from client.utils import d, l, CONFIG


class Session(Db, Ftp):
    def exist(self, id_or_ids):
        bulk = True
        if isinstance(id_or_ids, int):
            id_or_ids = [id_or_ids]
            bulk = False

        db_ids = []
        fs_ids = []
        for id in id_or_ids:
            if id > 0:
                db_ids.append(str(id))
            else:
                fs_ids.append(abs(id))

        if db_ids:
            rows = [row['id'] for row in
                    self.q("select id from blobs force index (id_idx) where id in (%s)" % ','.join(db_ids))]

        if fs_ids:
            if self.ftp is None:
                self.ftp = get_sftp()
            channel, transport = self.ftp

            for id in fs_ids:
                id_as_str = str(id).rjust(10, '0')
                path = join(CONFIG['ftp']['folder'], id_as_str[0:3], id_as_str[3:6], id_as_str[6:10])
                try:
                    channel.stat(path)
                    rows.append(-id)
                except IOError:
                    pass
        if bulk:
            return set(rows)
        else:
            if rows:
                return True
            else:
                return False

    def get_id(self, id_or_ids, default=None, order=False):
        bulk = True
        if isinstance(id_or_ids, int) or isinstance(id_or_ids, str):
            id_or_ids = [id_or_ids]
            bulk = False

        db_ids = []
        fs_ids = []
        for id in id_or_ids:
            id = int(id)
            if id > 0:
                db_ids.append(str(id))
            else:
                fs_ids.append(abs(id))

        tmp = {}
        if db_ids:
            rows = self.q(
                "select id, uncompress(val) val from blobs force index (id_idx) where id in (%s)" % ','.join(db_ids))
            for row in rows:
                tmp[row['id']] = l(row['val'])

        for id in fs_ids:
            id_as_str = str(id).rjust(10, '0')
            val = self.read(join(CONFIG['ftp']['folder'], id_as_str[0:3], id_as_str[3:6], id_as_str[6:10]))
            if val is not None:
                tmp[-id] = l(val)

        if not bulk and tmp:
            return list(tmp.values())[0]
        elif order:
            result = []
            for id in id_or_ids:
                if id in tmp:
                    result.append((id, tmp[id]))
            return OrderedDict(result)
        else:
            return tmp

    def q(self, *args, **kwargs):
        if 'load' in kwargs:
            load = kwargs['load']
            del kwargs['load']
            result = super(Session, self).q(*args, **kwargs)
            locations = [row[load] for row in result]
            values = self.get_id(locations)
            for idx in range(len(result)):
                result[idx][load] = values.get_id(result[idx][load])
            return result
        else:
            return super(Session, self).q(*args, **kwargs)

    def insert(self, value_or_values, bulk=False):
        if not bulk:
            value_or_values = [value_or_values]

        to_db = []
        to_fs = []
        markers = []
        for value in value_or_values:
            value_as_bytes = d(value)
            if sys.getsizeof(value_as_bytes) > 256 * 1024:
                to_fs.append(value_as_bytes)
                markers.append(True)
            else:
                to_db.append(value_as_bytes)
                markers.append(False)

        fs_ids = []
        if to_fs:
            max_id_on_ftp = self.q("select max_id_on_ftp from config")
            if not max_id_on_ftp:
                self.q("insert into config(max_id_on_ftp) values (0)")
                max_id_on_ftp = 0
            else:
                max_id_on_ftp = max_id_on_ftp[0]['max_id_on_ftp']

            while True:
                self.q("update config set max_id_on_ftp=%s where max_id_on_ftp=%s" % (
                max_id_on_ftp + len(to_fs), max_id_on_ftp))
                if self.cursor.rowcount > 0:
                    break
                else:
                    max_id_on_ftp = self.q("select max_id_on_ftp from config")[0]['max_id_on_ftp']

            for value_as_bytes, succ in zip(to_fs, range(1, len(to_fs) + 1)):
                id = max_id_on_ftp + succ
                id_as_str = str(id).rjust(10, '0')
                self.write(value_as_bytes, CONFIG['ftp']['folder'],
                           join(id_as_str[0:3], id_as_str[3:6], id_as_str[6:10]))
                fs_ids.append(-id)

        db_ids = []
        if to_db:
            last_id = self.q("insert into blobs(val) values %s" % ','.join(len(to_db) * ['(compress(%s))']), args=to_db, id=True)
            db_ids = list(range(last_id, last_id + len(to_db)))

        result = []
        for is_fs in markers:
            if is_fs:
                result.append(fs_ids.pop(0))
            else:
                result.append(db_ids.pop(0))

        if not bulk:
            return result[0]
        else:
            return result

    def delete(self, id_or_ids):
        if isinstance(id_or_ids, int) or isinstance(id_or_ids, str):
            id_or_ids = [id_or_ids]

        fs = []
        db = []
        for id in id_or_ids:
            id = int(id)
            if id < 0:
                fs.append(str(abs(id)))
            else:
                db.append(str(id))

        if db:
            self.q("delete from blobs where id in (%s)" % ','.join(db))

        paths = []
        for id in fs:
            id_as_str = id.rjust(10, '0')
            paths.append(
                join(CONFIG['ftp']['folder'], id_as_str[0:3], id_as_str[3:6], id_as_str[6:10])
            )
        self.delete_fs(paths)

    def size(self, id_or_ids):
        if isinstance(id_or_ids, int):
            id_or_ids = [id_or_ids]

        db = []
        fs = []
        for id in id_or_ids:
            if id > 0:
                db.append(str(id))
            else:
                fs.append(abs(id))

        result = {}
        if db:
            for row in self.q(
                    "select id, length(val) size from blobs force index (id_idx) where id in (%s)" % ','.join(db)):
                result[row['id']] = row['size']

        if fs:
            if self.ftp is None:
                self.ftp = get_sftp()

            channel, transport = self.ftp
            for id in fs:
                id_as_str = str(id).rjust(10, '0')
                result[-id] = channel.stat(
                    join(CONFIG['ftp']['folder'], id_as_str[0:3], id_as_str[3:6], id_as_str[6:10])).st_size
        return result

    def __exit__(self, *exc):
        if self.ftp is not None:
            channel, transport = self.ftp
            channel.close()
            transport.close()

        if self.db is not None:
            self.db.close()