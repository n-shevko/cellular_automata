import json
import pickle
import os
import paramiko

from typing import Dict, Any
from datetime import datetime, timedelta, date


def timedelta_to_dict(obj: timedelta) -> Dict[str, int]:
    return {
        'case': 'timedelta',
        'days': obj.days,
        'seconds': obj.seconds,
        'microseconds': obj.microseconds
    }


def datetime_to_dict(obj: datetime) -> Dict[str, int]:
    return {
        'case': 'datetime',
        'year': obj.year,
        'month': obj.month,
        'day': obj.day,
        'hour': obj.hour,
        'minute': obj.minute,
        'second': obj.second,
        'microsecond': obj.microsecond,
    }


def date_to_dict(obj: date) -> Dict[str, int]:
    return {
        'case': 'date',
        'year': obj.year,
        'month': obj.month,
        'day': obj.day
    }


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return datetime_to_dict(obj)
        if isinstance(obj, timedelta):
            return timedelta_to_dict(obj)
        if isinstance(obj, date):
            return date_to_dict(obj)
        return json.JSONEncoder.default(self, obj)


def json_to_obj(dct):
    if not isinstance(dct, dict):
        return dct
    type_name = dct.get('case')
    if not type_name:
        return dct
    elif type_name == 'datetime':
        del dct['case']
        return datetime(**dct)
    elif type_name == 'timedelta':
        del dct['case']
        return timedelta(**dct)
    elif type_name == 'date':
        del dct['case']
        return date(**dct)
    else:
        raise Exception("Can't convert from json to object")


def loads(value: str) -> Any:
    return json.loads(value, encoding='utf8', object_hook=json_to_obj)


def dumps(value: Any, **kwargs) -> str:
    kwargs['cls'] = CustomEncoder
    return json.dumps(value, **kwargs)


def l(obj: bytes) -> Any:
    if obj is None:
        return None
    else:
        return pickle.loads(obj)


def d(obj: Any) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def to_db_key(value: Any) -> str:
    return dumps(value, sort_keys=True, cls=CustomEncoder)


def get_config():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config.json'
    )
    with open(path, 'r') as f:
        config = json.loads(f.read())
    return config


ORIG = get_config()
use = ORIG.get('use')
if not use:
    if os.path.exists('/home/nikos/seafile'):
        use = 'local'
    else:
        use = 'remote'
CONFIG = ORIG[use]
PKEY = paramiko.RSAKey.from_private_key(open(CONFIG['pkey']))