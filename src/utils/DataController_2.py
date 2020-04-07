import pymysql
import pandas as pd
import time
import numpy as np

def get_time():
    return time.time()

def get_elapsed(start):
    end = time.time()
    elapsed = end - start
    print("Elapsed time: {:.1f} seconds | ~{:.1f} minutes | ~{:.1f} hours ".format(elapsed, elapsed/60, elapsed/3600))

def string_to_nparray(string_egm):
    a = string_egm.split()  # split string to list
    b = list(map(float, a))  # convert string elements to float elements
    egm = np.asarray(b)  # convert list to numpy array
    return egm


'''
# db = DatabaseController('root','','tachy','localhost')
# db = DatabaseController('root','','tachy','localhost')
# db.df = db.pd_read_sql('select * from data;')
# print(db.df.head())
'''
class DatabaseController:
    def __init__(self, usrname, pwd, db, host):
        self._conn = pymysql.connect(user=usrname, password=pwd, database=db, host=host)
        self._cursor = self._conn.cursor()
        self.df = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.commit()
        self.connection.close()

    @property
    def connection(self):
        return self._conn

    @property
    def cursor(self):
        return self._cursor

    def commit(self):
        self.connection.commit()

    # Executing non select sql statements
    def execute(self, sql, debug=None):
        with self._conn:
            self._cursor.execute(sql)
            if debug is not None:
                print("query executed: \n", sql)

    def fetchone(self, sql):
        with self._conn:
            self._cursor.execute(sql)
            row = self._cursor.fetchone()
            return row
            

    # Executing select sql statements - returns pd.DataFrame
    def pd_read_sql(self, sql):
        df = pd.read_sql(sql, self._conn)
        return df



