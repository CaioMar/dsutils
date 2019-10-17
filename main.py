import querybuilder as qb
import json

with open('./config.json','rb') as file:
    config = json.load(fp=file)

safras = ['201901']
query_dict = config["123456"]

print(qb.ConfigQueryReader(safras, query_dict)\
.build())

# print(qb.ConfigQueryReader(safra, query_dict).build())
#
# print(qb.ConfigQueryReader(safra, query_dict).build())
# test = qb.ConfigQueryReader(safra, query_dict).build()
print(type([1])==list)

print({'a':1})
