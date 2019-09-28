import querybuilder as qb
import json

modelos = \
{"123456":{
"tabela1_{}":{
"MAIN":True,
"ALIAS":"tb1_{}",
"DATABASE":"db",
"VARS":{
"var1_{}":{"SAFRAS":[0,1],"DEFASAGEM":0},
"var2_{}":{"SAFRAS":[0],"DEFASAGEM":0}
}
},
"tabela2_{}":{
"ALIAS":"tb2_{}",
"DATABASE":"db",
"VARS":{
"var3_{}":{"SAFRAS":[0],"DEFASAGEM":1},
"var4_{}":{"SAFRAS":[0,3],"DEFASAGEM":0}
}
}
}}


with open('./config.json','w') as file:
    json.dump(fp=file, obj=modelos, indent=4)


# with open('./config.json','rb') as file:
#     config2 = json.load(fp=file)
#
# class SelectBuilder:
#     """docstring for SelectBuilder."""
#
#     def __init__(self):
#         self.
#
# print(config2['modelo1'].keys())
