# import torch
#
# pth = r'net_092.pth'
# sta_dic = torch.load(pth)
# print('.pth type:', type(sta_dic))
# print('.pth len:', len(sta_dic))
# print('--------------------------')
# for k, i in sta_dic.keys(),sta_dic.values():
#     print(k, type(sta_dic[k]), sta_dic[k].shape)
#     print(i, type(sta_dic[i]), sta_dic[i].shape)
#     print('------------------分割线---------------')

import torch
import torchvision.models as models
pthfile = r'net_092.pth'
net = torch.load(pthfile,map_location=torch.device('cpu'))
print(net)