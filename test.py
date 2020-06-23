import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtSql import QSqlDatabase, QSqlQuery
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
import interface
import ctypes
from PIL import ImageEnhance
from playsound import playsound
import add
import addimage
import torch
from torch import nn
import cv2
import time
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from ResNet18 import ResNet18
import torch.nn.functional as F
from PIL import Image

count=0
savepath='features_pic'
transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class ft_net(nn.Module):

    def __init__(self):
        super(ft_net, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_ft = ResNet18().to(device)#models.resnet18(pretrained=True)
        self.model = model_ft

    # 绘制特征图
    def draw_features(self,width, height, x, savename):
        tic = time.time()
        fig = plt.figure(figsize=(16, 16))#将每张图片变成16*16尺寸
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)#left,bottom,right,top表示整张图的左下右上的白色空隙，wspace和hspace分别表示子图之间左右、上下的间距
        for i in range(width * height):
            plt.subplot(height, width, i + 1)#height*height的矩阵图中中的第i+1个
            plt.axis('off')
            # plt.tight_layout()
            img = x[0, i, :, :]#绘制第i维的图
            pmin = np.min(img)
            pmax = np.max(img)
            img = (img - pmin) / (pmax - pmin + 0.000001)#归一化[0,1]之间
            plt.imshow(img, cmap='gray')
            # print("{}/{}".format(i,width*height))
        fig.savefig(savename, dpi=100)#指定分辨率为100
        fig.clf()#清除所有轴
        plt.close()#关闭窗口
        print("time:{}".format(time.time() - tic))#计算时间

    def forward(self, x):
        if True: # draw features or not
            x = self.model.conv1(x)
            self.draw_features(8,8,x.cpu().numpy(),"{}/f1_conv1.png".format(savepath))#将tensor转化为numpy

            # x = self.model.bn1(x)
            # draw_features(8, F8, x.cpu().numpy(),"{}/f2_bn1.png".format(savepath))
            #
            # x = self.model.relu(x)
            # draw_features(8, 8, x.cpu().numpy(), "{}/f3_relu.png".format(savepath))

            # x = self.model.maxpool(x)
            # draw_features(8, 8, x.cpu().numpy(), "{}/f4_maxpool.png".format(savepath))

            x = self.model.layer1(x)#[1,64,32,32]
            self.draw_features(8, 8, x.cpu().numpy(), "{}/f2_layer1.png".format(savepath))

            x = self.model.layer2(x)#[1,128,16,16]
            self.draw_features(8, 16, x.cpu().numpy(), "{}/f3_layer2.png".format(savepath))

            x = self.model.layer3(x)#[1,256,8,8]
            self.draw_features(16, 16, x.cpu().numpy(), "{}/f4_layer3.png".format(savepath))

            x = self.model.layer4(x)#[1,512,4,4]
            self.draw_features(16, 16, x.cpu().numpy()[:, 0:256, :, :], "{}/f5_layer4_1.png".format(savepath))#先提取前256维特征
            self.draw_features(16, 16, x.cpu().numpy()[:, 256:512, :, :], "{}/f5_layer4_2.png".format(savepath))#再提取后256维特征

            # x = self.model.avgpool(x)
            x=F.avg_pool2d(x,4)#每2*2区域中计算一个平均值[1,512,1,1]
            plt.plot(np.linspace(1, 512, 512), x.cpu().numpy()[0, :, 0, 0])#[1,512]之间取512个数,前一个参数表示横坐标，后一个参数表示纵坐标
            plt.savefig("{}/f6_avgpool.png".format(savepath))
            plt.clf()
            plt.close()

            x = x.view(x.size(0), -1)#[1,512]
            x = self.model.fc(x)#[1,4]
            plt.plot(np.linspace(1, 4, 4), x.cpu().numpy()[0, :])
            plt.savefig("{}/f7_fc.png".format(savepath))
            plt.clf()
            plt.close()
        else :
            x = self.model.conv1(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.model.fc(x)

        return x

# 文字添加到数据库
class addgarbage(QWidget, add.Ui_Form):
    def __init__(self, parent=None):
        super(addgarbage, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.addcontent)
        self.pushButton.clicked.connect(self.close)
    def addcontent(self):
        name=self.lineEdit.text()
        category=self.lineEdit_2.text()
        if category=='可回收垃圾':
            category='1'
        elif category=='有害垃圾':
            category='2'
        elif category=='厨余垃圾':
            category='3'
        else:
            category='4'
        query = QSqlQuery()  # 实例化对象
        query.exec_("insert into rubbish(c_name,categoryId) values('"+name+"',"+category+")")  # 执行sql语句

# 图片添加到本地文件
class addpicture(QWidget, addimage.Ui_Form):
    def __init__(self, parent=None):
        super(addpicture, self).__init__(parent)
        self.setupUi(self)

        show = cv2.resize(tempimage, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        self.showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                      QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.imageshowlabel.setPixmap(QtGui.QPixmap.fromImage(self.showImage))  # 往显示视频的Label里 显示QImage
        self.harmButton.clicked.connect(self.funAddImage)
        self.kitchenButton.clicked.connect(self.funAddImage)
        self.otherButton.clicked.connect(self.funAddImage)
        self.recycleButton.clicked.connect(self.funAddImage)
    def funAddImage(self):
        pButton=self.sender()#获取信号源
        s=pButton.text()#获取信号源上的内容
        #保存到对应类别的文件夹中
        if s=='有害垃圾':
            cv2.imwrite('saveImg/H/'+str(count)+'.png',tempimage)
        elif s=='厨余垃圾':
            cv2.imwrite('saveImg/K/'+str(count)+'.png',tempimage)
        elif s=='其他垃圾':
            cv2.imwrite('saveImg/O/'+str(count)+'.png',tempimage)
        else:
            cv2.imwrite('saveImg/R/'+str(count)+'.png',tempimage)


class CameraMainWin(QtWidgets.QMainWindow,interface.Ui_MainWindow):


    def __init__(self):
        super(CameraMainWin, self).__init__()
        self.setupUi(self)

        self.tmpimage=0
        self.image=None

        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap=cv2.VideoCapture()#获取视频流
        self.CAM_NUM=1


        #隐藏文字查询中的部分部件
        self.defination_label.setVisible(False)
        self.spacing_label1.setVisible(False)
        self.spacing_label2.setVisible(False)
        self.content_label1.setVisible(False)
        self.content_label2.setVisible(False)

        #查找按钮信号&槽
        self.search_button.clicked.connect(self.search)


        # #设置槽函数（开始检测）
        self.pushButton.clicked.connect(self.takePhoto)

        #菜单栏中的事件处理（不同学习率的训练日志的显示）
        self.action0_1.triggered.connect(self.func0_1)
        self.action0_01.triggered.connect(self.func0_01)
        self.action0_001.triggered.connect(self.func0_001)
        self.feature_action.triggered.connect(self.func_show_feature)#特征图显示

        #连接数据库
        self.db_connect()

        #打开camera
        if self.timer_camera.isActive()==False:#若定时器未启动
            flag=self.cap.open(self.CAM_NUM)#打开摄像头
            if flag==False:#打开不成功
                msg=QtWidgets.QMessageBox.warning(self,'warning','请检查相机与电脑是否连接正确',buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(200)#定时器开始计时，结果是每过200ms从摄像头中取一帧显示
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.cameraLabel.clear()
        self.timer_camera.timeout.connect(self.show_camera)                                            # 若定时器结束，则调用show_camera()，每隔200ms调用show_camera函数

    def func0_1(self):
        self.Acc_label.setPixmap(QPixmap('img/Acc_0.1.png'))
        self.Loss_label.setPixmap(QPixmap('img/Loss_0.1.png'))
        f1 = open('txt/log0.1.txt', "r", encoding="utf-8")
        my_data1 = f1.read()
        f1.close()
        f2 = open('txt/Valacc0.1.txt', "r", encoding="utf-8")
        my_data2 = f2.read()
        f2.close()
        f3 = open('txt/Testacc0.1.txt', "r", encoding="utf-8")
        my_data3 = f3.read()
        f3.close()
        self.train_log_label.setText(my_data1)
        self.val_log_label.setText(my_data2)
        self.test_log_label.setText(my_data3)

    def func0_01(self):
        self.Acc_label.setPixmap(QPixmap('img/Acc_0.01.png'))
        self.Loss_label.setPixmap(QPixmap('img/Loss_0.01.png'))
        f1 = open('txt/log0.01.txt', "r", encoding="utf-8")
        my_data1 = f1.read()
        f1.close()
        f2 = open('txt/Valacc0.01.txt', "r", encoding="utf-8")
        my_data2 = f2.read()
        f2.close()
        f3 = open('txt/Testacc0.01.txt', "r", encoding="utf-8")
        my_data3 = f3.read()
        f3.close()
        self.train_log_label.setText(my_data1)
        self.val_log_label.setText(my_data2)
        self.test_log_label.setText(my_data3)

    def func0_001(self):
        self.Acc_label.setPixmap(QPixmap('img/Acc_0.001.png'))
        self.Loss_label.setPixmap(QPixmap('img/Loss_0.001.png'))
        f1 = open('txt/log0.001.txt', "r", encoding="utf-8")
        my_data1 = f1.read()
        f1.close()
        f2 = open('txt/Valacc0.001.txt', "r", encoding="utf-8")
        my_data2 = f2.read()
        f2.close()
        f3 = open('txt/Testacc0.001.txt', "r", encoding="utf-8")
        my_data3 = f3.read()
        f3.close()
        self.train_log_label.setText(my_data1)
        self.val_log_label.setText(my_data2)
        self.test_log_label.setText(my_data3)

    def func_show_feature(self):
        model=ft_net()#执行构造函数
        img = transform_test(self.tmpimage)#[3,32,32]
        img = img.unsqueeze(0)#[1,3,32,32]
        with torch.no_grad():
            start = time.time()#起始时间
            out = model(img)#训练模型，执行forward函数
            print("total time:{}".format(time.time() - start))
            # result = out.cpu().numpy()
            # ind = np.argsort(result, axis=1)  # 从小到大排序，将索引值赋给ind[0,2,3,1],最大的索引值为0
            # for i in range(4):
            #     print("predict:top {} = cls {} : score {}".format(i + 1, ind[0, 4 - i - 1], result[0, 4 - i - 1]))
            # print("done")

            self.labelfeature_1.setPixmap(QPixmap("features_pic/f1_conv1.png").scaled(256,256))
            self.labelfeature_2.setPixmap(QPixmap("features_pic/f2_layer1.png").scaled(256,256))
            self.labelfeature_3.setPixmap(QPixmap("features_pic/f3_layer2.png").scaled(256,256))
            self.labelfeature_4.setPixmap(QPixmap("features_pic/f4_layer3.png").scaled(256,256))
            self.labelfeature_5.setPixmap(QPixmap("features_pic/f5_layer4_1.png").scaled(256,256))
            self.labelfeature_6.setPixmap(QPixmap("features_pic/f5_layer4_2.png").scaled(256,256))
            self.labelfeature_7.setPixmap(QPixmap("features_pic/f6_avgpool.png").scaled(256,192))
            self.labelfeature_8.setPixmap(QPixmap("features_pic/f7_fc.png").scaled(256,192))
            self.labelfeature_8.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)

    def hisEqulColor2(self,img):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)#颜色空间转换，有BGR转化为YCrCb
        channels = cv2.split(ycrcb)

        # https://docs.opencv.org/4.1.0/d5/daf/tutorial_py_histogram_equalization.html
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe.apply(channels[0], channels[0])

        cv2.merge(channels, ycrcb)#合并通道
        cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)#将YCrCb转换为BGR
        return img

    def show_camera(self):
        flag, self.image = self.cap.read()  # 从视频流中读取，返回一个布尔值（读取帧正确返回True）和每一帧的图像
        # 增强对比度和亮度
        im = self.hisEqulColor2(self.image)
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))#实现array到image的转换

        enh1 = ImageEnhance.Contrast(im)
        im1 = enh1.enhance(1.1)  # 对比度增强1.1
        self.imarray = np.array(im1)#转化为array形式
        # global tempimage
        # tempimage=self.image
        show = cv2.resize(self.image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        self.showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.cameraLabel.setPixmap(QtGui.QPixmap.fromImage(self.showImage))

    def takePhoto(self):
        #定时器启动
        if self.timer_camera.isActive() != False:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            img = Image.fromarray(self.imarray.astype('uint8'))#实现array到image的转换
            self.tmpimage=img
            img = transform_test(img)#数据增强
            img = torch.unsqueeze(img, 0)#添加一个维度
            img = img.to(device)
            # 模型定义-ResNet
            net = ResNet18().to(device)
            net.load_state_dict(torch.load("net_092.pth"))

            with torch.no_grad():#禁止反向传播
                net.eval()#固定住权重参数，生成的model用来测试样本
                output = net(img)
                tmp = F.softmax(output, dim=1)#tmp是一个二维的数据
                tmp=tmp.squeeze(0)#减去一个维度

                confidence_harm=round(tmp[0].item()*100,2)
                confidence_kitchen = round(tmp[1].item()*100, 2)
                confidence_other = round(tmp[2].item()*100, 2)
                confidence_recycle = round(tmp[3].item()*100, 2)
                self.progressBar.setValue(confidence_harm)
                self.progressBar_2.setValue(confidence_kitchen)
                self.progressBar_3.setValue(confidence_other)
                self.progressBar_4.setValue(confidence_recycle)
                self.pro_label1.setText(str(round(tmp[0].item()*100,2))+"%")
                self.pro_label2.setText(str(round(tmp[1].item() * 100, 2))+"%")
                self.pro_label3.setText(str(round(tmp[2].item() * 100, 2))+"%")
                self.pro_label4.setText(str(round(tmp[3].item() * 100, 2))+"%")

                maxvalue=max(round(tmp[0].item()*100,2),round(tmp[1].item() * 100, 2),round(tmp[2].item() * 100, 2),round(tmp[3].item() * 100, 2))
                if maxvalue<=60:
                    global tempimage
                    tempimage = self.image
                    # playsound('sound/tips.mp3')
                    choice = QMessageBox.question(self, '提示', '加入数据集',
                                              QMessageBox.Yes | QMessageBox.No)

                    if choice == QMessageBox.Yes:
                        self.myaddpicture = addpicture()
                        self.myaddpicture.show()
                    elif choice == QMessageBox.No:
                        pass
                else:
                    _, predicted = torch.max(output.data, 1)
                    classes = ('harm', 'kitchen', 'other', 'recycle')
                    playsound('sound/'+classes[predicted]+'.mp3')
                    print(classes[predicted])
                    print(round(tmp[0].item() * 100, 2))
                    print(round(tmp[1].item() * 100, 2))
                    print(round(tmp[2].item() * 100, 2))
                    print(round(tmp[3].item() * 100, 2))

                    if classes[predicted]=='harm':
                        self.label.setPixmap(QtGui.QPixmap(":/pic/img/"+classes[predicted]+".png"))
                        self.label_2.setPixmap(QtGui.QPixmap(":/pic/img/" + "kitchen-before.png"))
                        self.label_3.setPixmap(QtGui.QPixmap(":/pic/img/" +  "other-before.png"))
                        self.label_4.setPixmap(QtGui.QPixmap(":/pic/img/" + "recycle-before.png"))
                    elif classes[predicted]=='kitchen':
                        self.label.setPixmap(QtGui.QPixmap(":/pic/img/" + "harm-before.png"))
                        self.label_2.setPixmap(QtGui.QPixmap(":/pic/img/"+classes[predicted]+".png"))
                        self.label_3.setPixmap(QtGui.QPixmap(":/pic/img/" +  "other-before.png"))
                        self.label_4.setPixmap(QtGui.QPixmap(":/pic/img/" + "recycle-before.png"))
                    elif classes[predicted] == 'other':
                        self.label.setPixmap(QtGui.QPixmap(":/pic/img/" + "harm-before.png"))
                        self.label_2.setPixmap(QtGui.QPixmap(":/pic/img/" + "kitchen-before.png"))
                        self.label_3.setPixmap(QtGui.QPixmap(":/pic/img/" + classes[predicted] + ".png"))
                        self.label_4.setPixmap(QtGui.QPixmap(":/pic/img/" + "recycle-before.png"))
                    else:
                        self.label.setPixmap(QtGui.QPixmap(":/pic/img/" + "harm-before.png"))
                        self.label_2.setPixmap(QtGui.QPixmap(":/pic/img/" + "kitchen-before.png"))
                        self.label_3.setPixmap(QtGui.QPixmap(":/pic/img/" + "other-before.png"))
                        self.label_4.setPixmap(QtGui.QPixmap(":/pic/img/"+classes[predicted]+".png"))


    #连接数据库
    def db_connect(self):
        ctypes.windll.LoadLibrary('E:/anzhuang/mysql-8.0.19-winx64/lib/libmysql.dll')
        self.db = QSqlDatabase.addDatabase('QMYSQL')
        self.db.setHostName('localhost')
        self.db.setDatabaseName('test_db')
        self.db.setUserName('root')
        self.db.setPassword('wyyhx123')
        if not self.db.open():
            QMessageBox.critical(self, 'Database Connection', self.db.lastError().text())

    #关闭数据库
    def closeEvent(self, QCloseEvent):
        self.db.close()

    # 显示消息提示框
    def show_messagebox(self):
        choice = QMessageBox.question(self, '提示', '未查询到，是否添加到数据库中',
                                      QMessageBox.Yes | QMessageBox.No)

        if choice == QMessageBox.Yes:
            self.myWin = addgarbage()
            self.myWin.show()
        elif choice == QMessageBox.No:
            pass
    #查询
    def search(self):
        text=self.search_edit.text()#获取文本框内容
        query=QSqlQuery()#实例化对象

        query.exec_("select rc_name,defination,c_explain,c_require from rubbish_category,rubbish"
                    " where c_name='"+text+"' and rubbish.categoryId=rubbish_category.id")#执行sql语句

        if query.next():#将记录指针定位到返回结果中的第一条（可能有多个返回值啦）
            rc_name=query.value(0)#传入响应索引值就可以返回指定的字段数据
            defination=query.value(1)
            c_explain=query.value(2)
            c_require=query.value(3)

            # 显示部件
            self.defination_label.setVisible(True)
            self.spacing_label1.setVisible(True)
            self.spacing_label2.setVisible(True)
            self.content_label1.setVisible(True)
            self.content_label2.setVisible(True)

            self.defination_label.setText(defination)
            self.spacing_label1.setText(rc_name+"主要包括")
            self.content_label1.setText(c_explain)
            self.spacing_label2.setText(rc_name + "投放要求")
            self.content_label2.setText(c_require)
        else:
            self.show_messagebox()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    W = CameraMainWin()
    # qss_spacing = "QLabel:{background-color:red}"
    # W.setStyleSheet(qss_spacing)
    W.show()
    sys.exit(app.exec_())