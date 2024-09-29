from Auto_scan_class import *
from multiprocessing import Manager
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from Autotest_V2 import Ui_MainWindow,    \
     imageThread,   tippathThread, fixtipThread


def create_shared_queues():
    manager = Manager()
    return manager.Queue(), manager.Queue(), manager.Queue(), manager.Queue()


# 主函数
if __name__ == '__main__':
    shared_queues = create_shared_queues()
    nanonis = Mustard_AI_Nanonis()
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = Ui_MainWindow(shared_queues, nanonis)  # 创建UI实例
    ui.setupUi(window)
    window.show()

    image_worker = imageThread(nanonis, ui)
    image_worker.finished.connect(app.quit)
    image_worker.start()

    tippathThread_worker = tippathThread(nanonis, ui)
    tippathThread_worker.finished.connect(app.quit)
    tippathThread_worker.start()

    tipfix_worker = fixtipThread(nanonis, shared_queues)
    tipfix_worker.finished.connect(app.quit)
    tipfix_worker.start()



    sys.exit(app.exec_())