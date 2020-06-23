# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'addimage.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(681, 550)
        self.layoutWidget = QtWidgets.QWidget(Form)
        self.layoutWidget.setGeometry(QtCore.QRect(0, 0, 681, 551))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setFocusPolicy(QtCore.Qt.NoFocus)
        self.label.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.label.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label.setLineWidth(0)
        self.label.setMidLineWidth(0)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setWordWrap(False)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.imageshowlabel = QtWidgets.QLabel(self.layoutWidget)
        self.imageshowlabel.setFrameShape(QtWidgets.QFrame.Box)
        self.imageshowlabel.setText("")
        self.imageshowlabel.setObjectName("imageshowlabel")
        self.verticalLayout.addWidget(self.imageshowlabel)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.harmButton = QtWidgets.QPushButton(self.layoutWidget)
        self.harmButton.setObjectName("harmButton")
        self.horizontalLayout.addWidget(self.harmButton)
        self.kitchenButton = QtWidgets.QPushButton(self.layoutWidget)
        self.kitchenButton.setObjectName("kitchenButton")
        self.horizontalLayout.addWidget(self.kitchenButton)
        self.otherButton = QtWidgets.QPushButton(self.layoutWidget)
        self.otherButton.setObjectName("otherButton")
        self.horizontalLayout.addWidget(self.otherButton)
        self.recycleButton = QtWidgets.QPushButton(self.layoutWidget)
        self.recycleButton.setObjectName("recycleButton")
        self.horizontalLayout.addWidget(self.recycleButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 3)
        self.verticalLayout.setStretch(2, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "请选择垃圾类别"))
        self.harmButton.setText(_translate("Form", "有害垃圾"))
        self.kitchenButton.setText(_translate("Form", "厨余垃圾"))
        self.otherButton.setText(_translate("Form", "其他垃圾"))
        self.recycleButton.setText(_translate("Form", "可回收垃圾"))


