import requests
import re
import os

word=input('请输入要下载图片的名称:')
# if not os.path.exists(word):
#     os.mkdir(word)

count=1
#1.拿到url
for k in range(0,1):
    url="https://image.baidu.com/search/flip?tn=baiduimage&istype=2&ie=utf-8&word="+word+"&pn="+str(k*20)

    #2.得到网页源代码
    r=requests.get(url)#利用requests这个模块来获取当前这个url的信息，得到数据
    #print(r)#<Response [200]> 200状态码，请求成功   Response对象
    ret=r.content.decode()#ret得到的就是网页源代码

    #3.拿到所有图片的url（链接地址）
    #"objURL":"",
    #正则表达式解释
    #.表示可以匹配除\n之外的任意字符，*表示可以匹配无数个（因为有多个objURL），？表示尽可能地少匹配
    result=re.findall('"objURL":"(.*?)",',ret)#result是一个列表，保存了图片的url

    #4.保存所有的图片
    for i in result:
        try:
            r=requests.get(i,timeout=1)#<Response [200]> 200状态码，请求成功   Response对象，timeout设置超时，如果1s后服务器不能得到响应，直接执行后面的代码
        except Exception as e:
            print(e)
            continue

        #判断url后几位是否是图片类型来结尾
        end=re.search('(/.jpg|/.jpeg|/.gif|/.png)$',i)#'$'表示以...终止,斜杠表示转义字符
        #如果不是，那么end就会是None
        temp=''
        if end==None:
            temp=str(count)+'.jpg'
        else:
            temp=str(count)+'.'+i.split('.')[-1]
        print(temp)
        count=count+1
        #创建一个文件，名字较‘path’,以w(write)b(bytes)方式写书
        with open('J:/temp/'+temp,'wb') as f:
            f.write(r.content)


