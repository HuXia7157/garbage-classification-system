import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import argparse
from ResNet18 import ResNet18
from PIL import ImageFile
import numpy as np
from visdom import Visdom
import numpy as np
np.set_printoptions(suppress=True, threshold=np.nan)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# to_pil_image  = transforms.ToPILImage()
unloader = transforms.ToPILImage()

#使用visdom显示图表
viz=Visdom()#新建一个连接客户端对象
viz2=Visdom()

#验证集和测试集的区别：验证集用于调整参数，当发生过拟合时，使得过拟合终止，如果不用
#验证集，直接根据测试集去调整参数，称不上真正意义上的测试，因为你是根据测试集的训练情况
#来调整参数，测试集仅仅用于测试。
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#如果gpu可用，则用gpu，否则用cpu

# 参数设置
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')#创建出解析器对象
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()#解析参数

# 参数设置
EPOCH = 130   #遍历数据集次数
pre_epoch = 120  # 上一个epoch数，便于加载模型
BATCH_SIZE = 50      #批处理尺寸(batch_size)
LR = 0.001        #学习率

# 准备数据集并预处理（数据扩充的一种方式）
transform_train = transforms.Compose([
    transforms.Resize(32),#保持图片宽高比，让更小的边变为32
    transforms.RandomCrop(32, padding=4),  #先四周填充0，再把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),#转化为tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差,计算公式：image=(image-mean)/std
])#前面一步是ToTensor(),所以在normalization之前会把[0,255]的值先转换到[0,1],normalize的作用是让数据满足正态分布，传给激活函数时数据变化明显，即梯度较大，学习速率更快

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在把图像随机裁剪成32*32
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#训练集
trainset=datasets.ImageFolder('train', transform_train)#加载数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
                                                                                                          #num_workers表示采用多线程方式导入数据，shuffle表示是否打乱数据
#验证集
valset=datasets.ImageFolder('validation', transform_test)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


#测试集
testset=datasets.ImageFolder('test', transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 模型定义-ResNet
net = ResNet18().to(device)#执行ResNet函数
#net.load_state_dict(torch.load("model/net_120.pth"))#加载模型

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题，真实概率分布与预测概率分布之间的差异。交叉熵的值越小，模型的预测效果越好
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减），权重衰减系数=lamda（不懂得话可以看一下思维导图）

#首先建立一个win,下面建立一个曲线的win,准确率
win=viz.line(
    X=np.array([3840]),
    Y=np.array([91.428]),
    opts=dict(title='Accuracy',legend=['train_Acc'],ytickmin=0,ytickmax=100),
    name="train_Acc"
)
#首先建立一个win,下面建立一个曲线的win，错误率
win2=viz2.line(
    X=np.array([3840]),#一维数组，数组中有两个元素
    Y=np.array([0.241]),
    opts=dict(title='Loss',legend=['train_Error'],ytickmin=0,ytickmax=2),
    name="train_Error"
)
#添加新的line到之前的win中去
viz.line(
    X=np.array([3840]),#一维数组，数组中有两个元素
    Y=np.array([86.995]),
    win=win,
    update="new",
    opts=dict(legend=['val_Acc']),
    name="val_Acc"
)
#添加新的line到之前的win中去
viz.line(
    X=np.array([3840]),#一维数组，数组中有两个元素
    Y=np.array([89.813]),
    win=win,
    update="new",
    opts=dict(legend=['test_Acc']),
    name="test_Acc"
)

#添加新的line到之前的win中去
viz2.line(
    X=np.array([3840]),#一维数组，数组中有两个元素
    Y=np.array([0.369]),
    win=win2,
    update="new",
    opts=dict(legend=['val_Error']),
    name="val_Error"
)
#添加新的line到之前的win中去
viz2.line(
    X=np.array([3840]),#一维数组，数组中有两个元素
    Y=np.array([0.298]),
    win=win2,
    update="new",
    opts=dict(legend=['test_Error']),
    name="test_Error"
)
# 训练
if __name__ == "__main__":
    best_acc = 85  #2 初始化best test accuracy
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数

    with open("Valacc.txt", "w") as f3:#打开文件
        with open("Testacc.txt", "w") as f:
            with open("log.txt", "w")as f2:
                for epoch in range(pre_epoch, EPOCH):#从pre_epoch+1开始到pre_epoch+50结束
                    print('\nEpoch: %d' % (epoch + 1))
                    net.train()#开始训练
                    sum_loss = 0.0
                    correct = 0.0
                    total = 0.0
                    for i, data in enumerate(trainloader, 0):#既获得索引也获得数据
                        # 准备数据
                        length = len(trainloader)#90(train中的数据集÷50，表示训练完整个数据集一共需要输入几次数据)
                        inputs, labels = data#input[50,3,32,32],label[50,1]

                        inputs, labels = inputs.to(device), labels.to(device)#拷贝一份数据到cpu上去
                        optimizer.zero_grad()#d_weight初始化为0，该变量用于记录同一变量的导数累加和

                        # forward + backward
                        outputs = net(inputs)#进行前向传播
                        loss = criterion(outputs, labels)
                        loss.backward()#反向传播
                        optimizer.step()#进行梯度下降，反向传播，更新权重参数

                        # 每训练1个batch打印一次loss和准确率
                        sum_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)#返回每一行最大的元素以及索引，output:[50,4
                        total += labels.size(0)
                        correct += predicted.eq(labels.data).cpu().sum()#tensor.eq:相等返回1，不等返回0
                        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                            % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct.item() / total))#loss:计算的是每个batch的平均loss
                        f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                            % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct.item() / total))
                        f2.write('\n')
                        f2.flush()
                        viz.line(
                            X=np.array([i + 1 + epoch * length]),
                            Y=np.array([100. * correct / total]),
                            win=win,
                            update="append",
                            name="train_Acc"
                        )
                        viz2.line(
                            X=np.array([i + 1 + epoch * length]),
                            Y=np.array([sum_loss / (i + 1)]),
                            win=win2,
                            update="append",
                            name="train_Error"
                        )
                    # 每训练完一个epoch测试一下准确率
                    print("Waiting Validation!")
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        val_loss=0
                        # list_val_acc=[]
                        for j,data in enumerate(valloader,0):
                            net.eval()#不使用dropout保持完整的模型，batchnorm用整个统计数据，
                            images, labels = data
                            images, labels = images.to(device), labels.to(device)
                            outputs = net(images)
                            loss=criterion(outputs,labels)
                            val_loss+=loss.item()
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum()
                        print('验证分类Loss为:%.03f,验证分类准确率为：%.3f%%' % (val_loss/len(valloader),100.0 * correct.item() / total))
                        acc = 100. * correct.item() / total
                        # 将每次测试结果实时写入acc.txt文件中
                        print('Saving model......')
                        torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                        f3.write("EPOCH=%03d,Loss=%.03f,Accuracy= %.3f%%" % (epoch + 1,val_loss/len(valloader), acc))
                        f3.write('\n')
                        f3.flush()
                        viz.line(
                            X=np.array([(epoch+1) * len(trainloader)]),
                            Y=np.array([100.0 * correct / total]),
                            win=win,
                            update="append",
                            name="val_Acc"
                        )
                        viz2.line(
                            X=np.array([(epoch+1) * len(trainloader)]),
                            Y=np.array([val_loss/len(valloader)]),
                            win=win2,
                            update="append",
                            name="val_Error"
                        )
                        #记录最佳测试分类准确率并写入best_acc.txt文件中
                        if acc > best_acc:
                            f4 = open("best_acc.txt", "w")
                            f4.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                            f4.close()
                            best_acc = acc

                    # 每训练完一个epoch测试一下准确率
                    print("Waiting test!")
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        tes_loss=0
                        for data in testloader:
                            net.eval()
                            images, labels = data
                            images, labels = images.to(device), labels.to(device)
                            outputs = net(images)
                            loss = criterion(outputs, labels)
                            tes_loss += loss.item()
                            # 取得分最高的那个类 (outputs.data的索引号)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum()
                        print('测试分类Loss为:%.03f,测试分类准确率为：%.3f%%' % (tes_loss / len(testloader), 100.0 * correct.item() / total))
                        acc = 100. * correct.item() / total
                        # 将每次测试结果实时写入acc.txt文件
                        f.write("EPOCH=%03d,Loss=%.03f,Accuracy= %.3f%%" % (epoch + 1, tes_loss / len(testloader), acc))
                        f.write('\n')
                        f.flush()
                        viz.line(
                            X=np.array([(epoch+1) * len(trainloader)]),
                            Y=np.array([100.0 * correct / total]),
                            win=win,
                            update="append",
                            name="test_Acc"
                        )
                        viz2.line(
                            X=np.array([(epoch+1) * len(trainloader)]),
                            Y=np.array([tes_loss / len(testloader)]),
                            win=win2,
                            update="append",
                            name="test_Error"
                        )

                        # 记录最佳测试分类准确率并写入best_acc.txt文件中
                        if acc > best_acc:
                            f4 = open("best_acc.txt", "w")
                            f4.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                            f4.close()
                            best_acc = acc
                print("Training Finished, TotalEPOCH=%d" % EPOCH)
