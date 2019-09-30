import json

modelos = \
{"123456":{
"tabela1_{}":{
"MAIN":0,
"ALIAS":"tb1_{}",
"DATABASE":"db",
"VARS":{
"var1_{}":{"SAFRAS":[0,1],"DEFASAGEM":0, "ALIAS":"var1_", "TRANSFORMATIONS":{1:[('=0',11),('=1',12),(15)]}},
"var2_{}":{"SAFRAS":[0],"DEFASAGEM":0, "ALIAS":"var2_", "TRANSFORMATIONS":{0:[('>=0','<=0',11),('>=1','<=2',12),(15)]}}
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



with open('./config2.json','w') as file:
    json.dump(fp=file, obj=modelos, indent=4)
