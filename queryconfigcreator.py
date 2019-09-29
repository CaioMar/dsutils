import json

modelos = \
{"123456":{
"tabela1_{}":{
"MAIN":0,
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
