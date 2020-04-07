import sys
sys.path.insert(1, '/home/sawsn/Shiernee/FHN/src/utils')

from DataController_2 import *

# cloud connection
# db = DatabaseController('b59b25a8f483fb', 'df9f008b', 'heroku_59f0bb8d5ef8c3d', 'us-cdbr-iron-east-05.cleardb.net')

# BII database connection
db = DatabaseController('admin', '4854197saw', 'diffusion_data', '172.20.192.178')

df = db.pd_read_sql("SELECT * FROM data WHERE caseid='21_52_36';")
egm = string_to_nparray(df['egm'][0])

print(df.head())