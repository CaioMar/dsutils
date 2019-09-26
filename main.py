import json

# with open('./config.json','w') as file:
#     json.dump(fp=file, obj=conf, indent=4)

with open('./config.json','rb') as file:
    config2 = json.load(fp=file)



print(1)


print(config2['modelo1'].keys())
