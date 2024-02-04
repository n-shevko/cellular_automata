import hashlib

from client.ids import Session as BaseSession
from client.utils import to_db_key


class Session(BaseSession):
    def get(self, key_or_keys, default_value=None, bulk=False):
        if not bulk:
            key_or_keys = [key_or_keys]

        tmp2 = []
        tmp3 = set()
        for key in key_or_keys:
            as_json = to_db_key(key)
            tmp3.add(as_json)
            md5 = hashlib.md5(as_json.encode('utf-8')).hexdigest()
            tmp2.append("'%s'" % md5)

        keys_ids = []
        values_ids = []
        for row in self.q(
                "select k, v from kv_storage force index (key_md5_idx) where key_md5 in (%s)" % ','.join(tmp2)):
            keys_ids.append(row['k'])
            values_ids.append(row['v'])

        data = self.get_id(keys_ids + values_ids)
        result = []
        for key_id, value_id in zip(keys_ids, values_ids):
            try:
                key = data[key_id]
                value = data[value_id]
            except KeyError:
                continue

            if key in tmp3:
                result.append(value)

        if not result:
            return default_value
        else:
            if bulk:
                return result
            else:
                return result[0]

    def remove(self, key_or_keys, bulk=False):
        if not bulk:
            key_or_keys = [key_or_keys]

        tmp2 = []
        tmp3 = set()
        for key in key_or_keys:
            as_json = to_db_key(key)
            tmp3.add(as_json)
            md5 = hashlib.md5(as_json.encode('utf-8')).hexdigest()
            tmp2.append("'%s'" % md5)

        keys = []
        values = []
        for row in self.q(
                "select k, v from kv_storage force index (key_md5_idx) where key_md5 in (%s)" % ','.join(tmp2)):
            keys.append(row['k'])
            values.append(row['v'])

        keys_as_objects = self.get_id(keys)
        values_to_drop = []
        keys_to_drop = []
        for key_id, value_id in zip(keys, values):
            try:
                key = keys_as_objects[key_id]
            except KeyError:
                keys_to_drop.append(str(key_id))
                values_to_drop.append(str(value_id))
                continue

            if key in tmp3:
                keys_to_drop.append(str(key_id))
                values_to_drop.append(str(value_id))

        self.delete(keys_to_drop + values_to_drop)
        if values_to_drop:
            self.q("delete from kv_storage where k in (%s)" % ','.join(keys_to_drop))

        return tmp2, tmp3

    def set(self, key_or_keys, value_or_values, bulk=False):
        md5_from_keys, keys_as_json = self.remove(key_or_keys, bulk=bulk)
        keys_as_json = list(keys_as_json)

        if not bulk:
            value_or_values = [value_or_values]

        ids = self.insert(keys_as_json + value_or_values, bulk=True)
        keys_ids = ids[:len(keys_as_json)]
        values_ids = ids[len(keys_as_json):]
        tmp = ["(%s, %s, %s)" % (md5, key_id, value_id) for md5, key_id, value_id in
               zip(md5_from_keys, keys_ids, values_ids)]
        self.q("insert into kv_storage(key_md5, k, v) values %s" % ','.join(tmp))

    def has(self, key_or_keys, bulk=False):
        if not bulk:
            key_or_keys = [key_or_keys]

        tmp2 = []
        tmp3 = {}
        for idx, key in zip(range(len(key_or_keys)), key_or_keys):
            as_json = to_db_key(key)
            tmp3[as_json] = idx
            md5 = hashlib.md5(as_json.encode('utf-8')).hexdigest()
            tmp2.append("'%s'" % md5)

        keys_ids = [row['k'] for row in
                    self.q("select k from kv_storage force index (key_md5_idx) where key_md5 in (%s)" % ','.join(tmp2))]
        keys = self.get_id(keys_ids)
        result = [False] * len(key_or_keys)
        for key in keys.values():
            idx = tmp3.get(key)
            if idx is not None:
                result[idx] = True

        if not bulk:
            return result[0]
        else:
            return result
