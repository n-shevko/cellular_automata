import time

from datetime import timedelta, datetime

from MySQLdb.cursors import DictCursor
from MySQLdb import OperationalError, connect

from client.utils import CONFIG


class DictCursorCust(DictCursor):
    _defer_warnings = True


class Db(object):
    db = None

    def __enter__(self):
        return self

    def q(self, query: str, args=None, id=False):
        if self.db is None:
            while True:
                try:
                    self.db = connect(
                        host=CONFIG['host'],
                        user=CONFIG['db']['user'],
                        passwd=CONFIG['db']['password'],
                        db=CONFIG['db']['db'],
                        port=CONFIG['db']['port'],
                        autocommit=True
                    )
                    break
                except OperationalError as e:
                    if e.args[0] == 1040:
                        print('Waiting for mysql ...')
                        time.sleep(1)
                    else:
                        raise e
            self.cursor = self.db.cursor(DictCursorCust)

        start = datetime.now()
        self.cursor.execute(query, args=args)
        result = list(self.cursor.fetchall())
        finish = datetime.now()
        delta = finish - start
        if delta > timedelta(seconds=5):
            print('Slow query %s: %s' % (delta, query))

        if id:
            return self.db.insert_id()
        else:
            return result

    def __exit__(self, *exc):
        if self.db is not None:
            self.db.close()
