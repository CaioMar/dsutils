"""
Author: Caio Martins
Since: 2019-12
"""
import json

import querybuilder as qb

with open('./config.json', 'rb') as file:
    CONFIG = json.load(fp=file)

SAFRAS = ['201901']
QUERY_DICT = CONFIG["123456"]

print(qb.ConfigQueryReader(SAFRAS, QUERY_DICT)\
.build())
