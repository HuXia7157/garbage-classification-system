#encoding:utf8
from aip import AipSpeech

""" APPID AK SK """
APP_ID = '19495033'
API_KEY = 'VIVfMRAygh6kGWkXv79tdOSN'
SECRET_KEY = 'NpXBB0db0kgYE8i9e01HhIXMNlotbL1a'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
result = client.synthesis(text = '该物品为其他垃圾', options={'vol':5})

if not isinstance(result,dict):#判断result对象是否为字典数据类型
    with open('sound/other.mp3','wb') as f:
        f.write(result)
else:print(result)


