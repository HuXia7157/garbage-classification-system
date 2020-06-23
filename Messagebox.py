import sys
from PyQt5.QtWidgets import QApplication, QWidget
import add # 导入生成的窗口 其中python_pyechrts_ui 是UI变更PY文件名


# 新建一个类来继承生成的窗口，也可以在这里添加关于窗口处理的代码
class addgarbage(QWidget, add.Ui_Form):
    def __init__(self, parent=None):
        super(addgarbage, self).__init__(parent)
        self.setupUi(self)


# 主程序，生成一个窗口实例并运行。
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = addgarbage()
    myWin.show()
    sys.exit(app.exec_())
