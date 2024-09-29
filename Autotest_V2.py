import time

from PyQt5.QtGui import QImage, QPixmap
from Auto_scan_class import *
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QWidget, QLabel, \
    QVBoxLayout, QInputDialog
from PyQt5 import QtWidgets


class Ui_MainWindow(QMainWindow):
    def __init__(self, shared_queues, nanonis, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.shared_queues = shared_queues  # 存储共享队列作为类属性
        self.setupUi(self)
        self.nanonis = nanonis
        self.worker_thread = None
    def setupUi(self, MainWindow):
        self.lockin_window = None
        MainWindow.setObjectName("MainWindow")
        # MainWindow.resize(790, 322)
        MainWindow.resize(1250, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)

        # 创建一个标签作为注释
        # self.modeLabel = QtWidgets.QLabel(self.centralwidget)
        # self.modeLabel.setText("Scan Model:")
        # self.modeLabel.setGeometry(QtCore.QRect(550, 10, 100, 20))
        # # 添加模式选择下拉菜单
        # self.modeComboBox = QtWidgets.QComboBox(self.centralwidget)
        # self.modeComboBox.setGeometry(QtCore.QRect(650, 10, 100, 22))
        # self.modeComboBox.setObjectName("modeComboBox")
        # self.modeComboBox.addItem("Auto")
        # self.modeComboBox.addItem("Manual")
        # Manual 模式特有的控件

        self.manualGroup = QtWidgets.QGroupBox(self.centralwidget)
        self.manualGroup.setGeometry(QtCore.QRect(10, 50, 300, 250))  # 调整大小和位置
        self.manualGroup.setTitle("Manual Mode")
        self.manualGroup.hide()  # 初始时隐藏 Manual 模式控件

        # 使用 QVBoxLayout 作为 main 布局
        manualGroupLayout = QtWidgets.QVBoxLayout(self.manualGroup)
        manualGroupLayout.setContentsMargins(10, 10, 10, 10)  # 设置外边距

        # 功能一：选择扫图数量并开始扫图
        self.scanControlForm = QtWidgets.QFormLayout()
        self.scanCountLabel = QtWidgets.QLabel("Scan Count:")
        self.scanCountSpinBox = QtWidgets.QSpinBox()
        self.startScanButton = QtWidgets.QPushButton("Start Scan")
        self.scanControlForm.addRow(self.scanCountLabel, self.scanCountSpinBox)
        self.scanControlForm.addWidget(self.startScanButton)

        # 功能二：Tipshaper按钮和选择扎针深度
        self.tipSharpForm = QtWidgets.QFormLayout()
        self.nDepthLabel = QtWidgets.QLabel("Tip Sharpener Depth:")
        self.nDepthSpinBox = QtWidgets.QDoubleSpinBox()
        self.tipSharperButton = QtWidgets.QPushButton("Tip Sharpener")
        self.tipSharpForm.addRow(self.nDepthLabel, self.nDepthSpinBox)
        self.tipSharpForm.addWidget(self.tipSharperButton)

        # 功能三：设置扫描图像大小 (scanSize)
        self.scanSizeForm = QtWidgets.QFormLayout()
        self.widthLabel = QtWidgets.QLabel("Width (nm):")
        self.widthSpinBox = QtWidgets.QDoubleSpinBox()
        self.heightLabel = QtWidgets.QLabel("Height (nm):")
        self.heightSpinBox = QtWidgets.QDoubleSpinBox()
        self.scanSizeForm.addRow(self.widthLabel, self.widthSpinBox)
        self.scanSizeForm.addRow(self.heightLabel, self.heightSpinBox)
        self.sizeapplyButton = QtWidgets.QPushButton('apply')
        self.scanSizeForm.addWidget(self.sizeapplyButton)
        # 将表单布局添加到主布局
        manualGroupLayout.addLayout(self.scanControlForm)
        manualGroupLayout.addLayout(self.tipSharpForm)
        manualGroupLayout.addLayout(self.scanSizeForm)

        # 调整控件属性
        self.widthSpinBox.setRange(10, 300)
        self.widthSpinBox.setSingleStep(10)
        self.heightSpinBox.setRange(10, 300)
        self.heightSpinBox.setSingleStep(10)
        self.nDepthSpinBox.setRange(-6.0, -1.5)
        self.nDepthSpinBox.setSingleStep(0.1)

        # 将 QVBoxLayout 设置给 manualGroup
        self.manualGroup.setLayout(manualGroupLayout)
        # Auto模式的控件归类到autoGroup中
        self.autoGroup = QtWidgets.QGroupBox(self.centralwidget)
        self.autoGroup.setGeometry(QtCore.QRect(10, 40, 1200, 730))
        self.autoGroup.setTitle("Auto Mode Settings")
        self.autoGroup.setObjectName("autoGroup")

        # 将原有的控件添加到autoGroup中
        self.label = QtWidgets.QLabel(self.autoGroup)
        self.label.setGeometry(QtCore.QRect(490, 40, 131, 41))
        self.label.setStyleSheet("font: 10pt \"Alibaba PuHuiTi\";")
        self.label.setObjectName("label")
        self.PulseButton = QtWidgets.QPushButton(self.autoGroup)
        self.PulseButton.setGeometry(QtCore.QRect(600, 100, 101, 31))
        self.PulseButton.setObjectName("PulseButton")
        self.label_4 = QtWidgets.QLabel(self.autoGroup)
        self.label_4.setGeometry(QtCore.QRect(130, 50, 101, 21))
        self.label_4.setStyleSheet("font: 10pt \"Alibaba PuHuiTi\";")
        self.label_4.setObjectName("label_4")
        self.line = QtWidgets.QFrame(self.autoGroup)
        self.line.setGeometry(QtCore.QRect(450, 170, 251, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")

        self.line_4 = QtWidgets.QFrame(self.autoGroup)
        self.line_4.setGeometry(QtCore.QRect(780, 150, 321, 16))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line")

        self.line_2 = QtWidgets.QFrame(self.autoGroup)
        self.line_2.setGeometry(QtCore.QRect(390, 60, 20, 171))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(self.autoGroup)
        self.line_3.setGeometry(QtCore.QRect(150, 90, 16, 141))
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.label_5 = QtWidgets.QLabel(self.autoGroup)
        self.label_5.setGeometry(QtCore.QRect(210, 150, 140, 16))
        self.label_5.setObjectName("label_5")
        self.widget = QtWidgets.QWidget(self.autoGroup)
        self.widget.setGeometry(QtCore.QRect(180, 170, 172, 30))
        self.widget.setObjectName("widget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.time = QtWidgets.QDoubleSpinBox(self.widget)
        self.time.setObjectName("time")
        self.horizontalLayout_4.addWidget(self.time)
        self.time_apply_button = QtWidgets.QPushButton(self.widget)
        self.time_apply_button.setObjectName("apply")
        self.horizontalLayout_4.addWidget(self.time_apply_button)

        self.label_bias = QtWidgets.QLabel(self.autoGroup)
        self.label_bias.setGeometry(QtCore.QRect(210, 100, 140, 16))
        self.label_bias.setObjectName("bias")
        self.widget_bias = QtWidgets.QWidget(self.autoGroup)
        self.widget_bias.setGeometry(QtCore.QRect(180, 110, 172, 30))
        self.widget_bias.setObjectName("widget")
        self.horizontalLayout_bias = QtWidgets.QHBoxLayout(self.widget_bias)
        self.horizontalLayout_bias.setContentsMargins(0, 10, 0, 0)
        self.horizontalLayout_bias.setObjectName("horizontalLayout_4")
        self.bias = QtWidgets.QDoubleSpinBox(self.widget_bias)
        self.bias.setObjectName("time")
        self.horizontalLayout_bias.addWidget(self.bias)
        self.bias_apply_button = QtWidgets.QPushButton(self.widget_bias)
        self.bias_apply_button.setObjectName("apply")
        self.horizontalLayout_bias.addWidget(self.bias_apply_button)

        self.layoutWidget1 = QtWidgets.QWidget(self.autoGroup)
        self.layoutWidget1.setGeometry(QtCore.QRect(450, 200, 251, 31))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.TipSharperButton = QtWidgets.QPushButton(self.layoutWidget1)
        self.TipSharperButton.setObjectName("TipSharperButton")
        self.horizontalLayout_3.addWidget(self.TipSharperButton)
        self.SharptipValue = QtWidgets.QDoubleSpinBox(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.SharptipValue.sizePolicy().hasHeightForWidth())
        self.SharptipValue.setSizePolicy(sizePolicy)
        self.SharptipValue.setObjectName("SharptipValue")
        self.SharptipValue.setMinimum(-6)
        self.SharptipValue.setMaximum(-1.5)
        self.SharptipValue.setSingleStep(0.1)
        self.horizontalLayout_3.addWidget(self.SharptipValue)
        self.splitter = QtWidgets.QSplitter(self.autoGroup)
        self.splitter.setGeometry(QtCore.QRect(10, 90, 101, 141))
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.splitter_2 = QtWidgets.QSplitter(self.splitter)
        self.splitter_2.setOrientation(QtCore.Qt.Vertical)
        self.splitter_2.setObjectName("splitter_2")
        self.widget = QtWidgets.QWidget(self.splitter_2)
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.StartButton = QtWidgets.QPushButton(self.widget)
        self.StartButton.setObjectName("StartButton")
        self.verticalLayout_2.addWidget(self.StartButton)
        # self.StopButton = QtWidgets.QPushButton(self.widget)
        # self.StopButton.setObjectName("StopButton")
        # self.verticalLayout_2.addWidget(self.StopButton)
        self.PauseButton = QtWidgets.QPushButton(self.widget)
        self.PauseButton.setObjectName("PauseButton")
        self.verticalLayout_2.addWidget(self.PauseButton)
        self.ResumeButton = QtWidgets.QPushButton(self.widget)
        self.ResumeButton.setObjectName("pushButton")
        self.verticalLayout_2.addWidget(self.ResumeButton)
        self.widget1 = QtWidgets.QWidget(self.autoGroup)
        self.widget1.setGeometry(QtCore.QRect(450, 90, 145, 61))
        self.widget1.setObjectName("widget1")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget1)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(self.widget1)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.BiasValue = QtWidgets.QDoubleSpinBox(self.widget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.BiasValue.sizePolicy().hasHeightForWidth())
        self.BiasValue.setSizePolicy(sizePolicy)
        self.BiasValue.setMaximumSize(QtCore.QSize(70, 24))
        self.BiasValue.setMinimum(-50.0)
        self.BiasValue.setMaximum(50.0)
        self.BiasValue.setObjectName("BiasValue")
        self.horizontalLayout.addWidget(self.BiasValue)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_3 = QtWidgets.QLabel(self.widget1)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.WidthValue = QtWidgets.QDoubleSpinBox(self.widget1)
        self.WidthValue.setMaximum(200)
        self.WidthValue.setSingleStep(10)
        self.WidthValue.setObjectName("WidthValue")
        self.horizontalLayout_2.addWidget(self.WidthValue)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.bias.setMinimum(-2)
        self.bias.setMaximum(2)
        self.bias.setSingleStep(0.1)
        self.bias.setValue(-1)

        self.Set_1 = QtWidgets.QPushButton(self.autoGroup)
        self.Set_1.setGeometry(QtCore.QRect(1000, 180, 93, 28))
        self.Set_1.setObjectName("Set_1")
        self.line_4 = QtWidgets.QFrame(self.autoGroup)
        self.line_4.setGeometry(QtCore.QRect(730, 60, 20, 171))
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.widget = QtWidgets.QWidget(self.autoGroup)
        self.widget.setGeometry(QtCore.QRect(780, 170, 209, 61))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.Proportional = QtWidgets.QLabel(self.widget)
        self.Proportional.setObjectName("Proportional")
        self.horizontalLayout_7.addWidget(self.Proportional)
        self.proportional_view = QtWidgets.QDoubleSpinBox(self.widget)
        self.proportional_view.setObjectName("proportional_view")
        self.horizontalLayout_7.addWidget(self.proportional_view)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.Integral = QtWidgets.QLabel(self.widget)
        self.Integral.setObjectName("Integral")
        self.horizontalLayout_8.addWidget(self.Integral)
        self.Integral_view = QtWidgets.QDoubleSpinBox(self.widget)
        self.Integral_view.setObjectName("Integral_view")
        self.horizontalLayout_8.addWidget(self.Integral_view)
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        self.widget1 = QtWidgets.QWidget(self.autoGroup)
        self.widget1.setGeometry(QtCore.QRect(780, 100, 283, 30))
        self.widget1.setObjectName("widget1")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.widget1)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")

        self.Setpoint = QtWidgets.QLabel(self.widget1)
        self.Setpoint.setObjectName("Setpoint")
        self.horizontalLayout_6.addWidget(self.Setpoint)
        self.setpoint_value = QtWidgets.QDoubleSpinBox(self.widget1)
        self.setpoint_value.setObjectName("setpoint_value")
        self.horizontalLayout_6.addWidget(self.setpoint_value)
        self.Set = QtWidgets.QPushButton(self.widget1)
        self.Set.setObjectName("Set")
        self.horizontalLayout_6.addWidget(self.Set)
        self.label_9 = QtWidgets.QLabel(self.autoGroup)
        self.label_9.setGeometry(QtCore.QRect(830, 50, 151, 20))
        self.label_9.setStyleSheet("font: 10pt \"Alibaba PuHuiTi\";")
        self.label_9.setObjectName("label_9")
        self.Frame_path = QtWidgets.QLabel(self.autoGroup)
        self.Frame_path.setGeometry(QtCore.QRect(891, 660, 128, 28))
        self.Frame_path.setObjectName("Frame save path:")
        self.Frame_path_button = QtWidgets.QPushButton(self.autoGroup)
        self.Frame_path_button.setGeometry(QtCore.QRect(1026, 660, 93, 28))
        self.Frame_path_button.setObjectName("open")
        self.frame_move_button = QtWidgets.QPushButton(self.autoGroup)
        self.frame_move_button.setGeometry(QtCore.QRect(1026, 570, 93, 28))
        self.frame_move_button.setObjectName("HOME")
        self.tipfix_button = QtWidgets.QPushButton(self.autoGroup)
        self.tipfix_button.setGeometry(QtCore.QRect(900, 570, 93, 28))
        self.tipfix_button.setObjectName("Move")
        self.Home_button = QtWidgets.QPushButton(self.autoGroup)
        self.Home_button.setGeometry(QtCore.QRect(1026, 600, 93, 28))
        self.Home_button.setObjectName("HOME")
        self.Move_button = QtWidgets.QPushButton(self.autoGroup)
        self.Move_button.setGeometry(QtCore.QRect(900, 600, 93, 28))
        self.Move_button.setObjectName("Move")
        self.XA = QtWidgets.QLabel(self.autoGroup)
        self.XA.setGeometry(QtCore.QRect(891, 630, 128, 28))
        self.XA.setObjectName("Lockin-X(A)")
        self.XA_button = QtWidgets.QPushButton(self.autoGroup)
        self.XA_button.setGeometry(QtCore.QRect(1026, 630, 93, 28))
        self.XA_button.setObjectName("XA")
        self.Z = QtWidgets.QLabel(self.autoGroup)
        self.Z.setGeometry(QtCore.QRect(891, 690, 128, 28))
        self.Z.setObjectName("Z and current:")
        self.z_button = QtWidgets.QLabel('计数: 0', self.autoGroup)
        self.z_button.setGeometry(QtCore.QRect(1050, 690, 50, 28))
        self.z_button.setObjectName("count")
        self.setpoint_value.setMaximum(200)
        self.setpoint_value.setSingleStep(10)
        self.proportional_view.setMaximum(12)
        self.proportional_view.setMinimum(0)
        self.proportional_view.setSingleStep(1)
        self.Integral_view.setMaximum(120)
        self.Integral_view.setMinimum(0)
        self.Integral_view.setSingleStep(10)
        self.WidthValue.setValue(50)
        self.BiasValue.setValue(-4)
        self.time.setValue(90)
        self.setpoint_value.setValue(100)
        self.proportional_view.setValue(8)
        self.Integral_view.setValue(80)


        self.image_for_view = QtWidgets.QGraphicsView(self.autoGroup)
        self.image_for_view.resize(400, 400)
        self.image_for_view.move(20, 300)
        self.image_for = QtWidgets.QLabel(self.autoGroup)
        self.image_for.resize(400, 400)
        self.image_for.move(20, 300)

        self.image_for_label = QtWidgets.QLabel(self.autoGroup)
        self.image_for_label.resize(160, 20)  # 设置 QLabel 的大小
        self.image_for_label.move(30, 260)  # 设置 QLabel 的位置

        self.image_back_view = QtWidgets.QGraphicsView(self.autoGroup)
        self.image_back_view.resize(400, 400)  # 设置 QLabel 的大小
        self.image_back_view.move(450, 300)  # 设置 QLabel 的位置
        self.image_back = QtWidgets.QLabel(self.autoGroup)
        self.image_back.resize(400, 400)
        self.image_back.move(450, 300)

        self.image_back_label = QtWidgets.QLabel(self.autoGroup)
        self.image_back_label.resize(160, 20)
        self.image_back_label.move(460, 260)

        self.image_tip_path_view = QtWidgets.QGraphicsView(self.autoGroup)
        self.image_tip_path_view.resize(256, 256)
        self.image_tip_path_view.move(880, 300)
        self.image_tip_path = QtWidgets.QLabel(self.autoGroup)
        self.image_tip_path.resize(256, 256)
        self.image_tip_path.move(880, 300)

        self.image_tip_path_label = QtWidgets.QLabel(self.autoGroup)
        self.image_tip_path_label.resize(160, 20)
        self.image_tip_path_label.move(880, 260)

        self.contrast = QtWidgets.QSlider(self.autoGroup)
        self.contrast.setGeometry(QtCore.QRect(230, 260, 160, 22))
        self.contrast.setOrientation(QtCore.Qt.Horizontal)
        self.contrast.setObjectName("count")
        self.contrast.setMinimum(25)
        self.contrast.setMaximum(300)
        self.contrast.setValue(100)

        self.contrast.valueChanged.connect(self.contrast_changed)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1072, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        # self.PulseButton.clicked.connect(self.on_pulse_clicked)
        self.PulseButton.clicked.connect(self.pulse)
        self.TipSharperButton.clicked.connect(lambda: self.queue_tipsharp(self.SharptipValue.value()))
        self.time_apply_button.clicked.connect(self.on_apply_button_clicked)
        self.PauseButton.clicked.connect(lambda: self.scan_control(2))
        # self.StopButton.clicked.connect(lambda: self.queue_controls(1))
        self.ResumeButton.clicked.connect(lambda: self.scan_control(3))
        # self.modeComboBox.currentIndexChanged.connect(self.on_mode_changed)
        self.StartButton.clicked.connect(self.show_choice_dialog)
        self.Set.clicked.connect(self.queue_setpoint)
        self.Set_1.clicked.connect(self.on_set_clicked)
        self.Frame_path_button.clicked.connect(self.selectFolder)
        # self.z_button.clicked.connect(self.onStartPlot)
        self.Home_button.clicked.connect(self.home_set)
        self.XA_button.clicked.connect(self.Lock_in_XA_get)
        self.Move_button.clicked.connect(self.move_area)
        self.frame_move_button.clicked.connect(self.frame_move)
        self.tipfix_button.clicked.connect(self.tipfix)
        self.bias_apply_button.clicked.connect(self.bias_set)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_good_image_count)
        self.timer.start(2000)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Autoscan"))
        self.label.setText(_translate("MainWindow", "Bias Pulse"))
        self.PulseButton.setText(_translate("MainWindow", "Pulse"))
        self.label_4.setText(_translate("MainWindow", "Scan Control"))
        self.time_apply_button.setText(_translate("MainWindow", "apply"))
        self.TipSharperButton.setText(_translate("MainWindow", "TipSharper(n)"))
        self.StartButton.setText(_translate("MainWindow", "Start"))
        # self.StopButton.setText(_translate("MainWindow", "Stop"))
        self.PauseButton.setText(_translate("MainWindow", "Pause"))
        self.ResumeButton.setText(_translate("MainWindow", "Resume"))
        self.label_2.setText(_translate("MainWindow", "Bias(V)"))
        self.label_3.setText(_translate("MainWindow", "Width(ms)"))
        self.manualGroup.setTitle(_translate("MainWindow", "Manual Settings"))
        self.nDepthLabel.setText(_translate("MainWindow", "n Depth:"))
        self.scanCountLabel.setText(_translate("MainWindow", "Scan Count:"))
        self.label_5.setText(_translate("MainWindow", "Time-per-Frame(s)"))
        self.image_for_label.setText(_translate("MainWindow", "Image_for:"))
        self.image_back_label.setText(_translate("MainWindow", "Image_back:"))
        self.image_tip_path_label.setText(_translate("MainWindow", "Tip path:"))
        self.Set_1.setText(_translate("MainWindow", "Set"))
        self.Proportional.setText(_translate("MainWindow", "Proportional(pm)"))
        self.Integral.setText(_translate("MainWindow", "Integral(nm/s)"))
        self.Setpoint.setText(_translate("MainWindow", "Setpoint(pA):"))
        self.Set.setText(_translate("MainWindow", "Set"))
        self.label_9.setText(_translate("MainWindow", "Z-Controller"))
        self.Frame_path.setText(_translate("MainWindow", "Frame save path:"))
        self.Frame_path_button.setText(_translate("MainWindow", "open"))
        self.Z.setText(_translate("MainWindow", "good images :"))
        self.z_button.setText(_translate("MainWindow", "0"))
        self.Home_button.setText(_translate("MainWindow", "Home"))
        self.XA_button.setText(_translate("MainWindow", "get"))
        self.XA.setText(_translate("MainWindow", "Lockin-X(A):"))
        self.Move_button.setText(_translate("MainWindow", "Tip_move"))
        self.tipfix_button.setText(_translate("MainWindow", "Tipfix"))
        self.frame_move_button.setText(_translate("MainWindow", "Frame_move"))
        self.bias_apply_button.setText(_translate("MainWindow", "Set"))
        self.label_bias.setText(_translate("MainWindow", "Bias(V)"))
    # def a function to change contrast
    def contrast_changed(self, value):
        if self.nanonis.scale_signal == 1:
            alpha = value*0.01
            image_for = self.nanonis.scale_image_for
            image_back = self.nanonis.scale_image_back
            adjusted_for = cv2.convertScaleAbs(image_for, alpha=alpha)
            adjusted_back = cv2.convertScaleAbs(image_back, alpha=alpha)
            image_for = np.array(adjusted_for, dtype=np.uint8)
            image_back = np.array(adjusted_back, dtype=np.uint8)
            qt_image_for = QImage(
                image_for.data,
                image_for.shape[1],  # 宽度
                image_for.shape[0],  # 高度
                image_for.strides[0],  # 每行的字节数
                QImage.Format_Grayscale8  # 使用 QImage 的类属性而不是 QtGui 的
            )
            qt_image_back = QImage(
                image_back.data,
                image_back.shape[1],  # 宽度
                image_back.shape[0],  # 高度
                image_back.strides[0],  # 每行的字节数
                QImage.Format_Grayscale8  # 使用 QImage 的类属性而不是 QtGui 的
            )
            pixmap_for = QPixmap.fromImage(qt_image_for)
            pixmap_back = QPixmap.fromImage(qt_image_back)
            scaled_pixmap_for = pixmap_for.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scaled_pixmap_back = pixmap_back.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_for.setPixmap(scaled_pixmap_for)
            self.image_for.show()
            self.image_back.setPixmap(scaled_pixmap_back)
            self.image_back.show()
        else: pass
    def update_good_image_count(self):
        self.z_button.setText(f'{self.nanonis.good_image_count}')
    #     set bias
    def bias_set(self):
        bias = self.bias.value()
        self.nanonis.BiasSet(bias)
    #     auto tip fix thread
    def tipfix(self):
        tipfix = self.shared_queues[2]
        tipfix.put(1)
        message_box = QMessageBox()
        message_box.setWindowTitle("Fix-tip")
        message_box.setIcon(QMessageBox.Question)
        message_box.setText("tip fixing......")
        stop_button = message_box.addButton("Finish", QMessageBox.AcceptRole)
        stop_button.setText("Finish")
        message_box.exec_()

        # 根据用户的选择执行相应操作
        if message_box.clickedButton() == stop_button:
            print("fix stop.")
            stop_signal = self.shared_queues[3]
            stop_signal.put(0)
    # def withdraw(self):
    #     self.nanonis.ScanStop()
    #     self.nanonis.ScanFrameSet(0, 0, 5E-8, 5E-8)
    #     self.nanonis.ZCtrlWithdraw(1)
    #     self.nanonis.MotorMoveSet('Z-', 30000)
    # define a signal to move frame
    def frame_move(self):
        if self.nanonis.line_scan_signal == 0:
            self.nanonis.frame_move_signal = 1
    # a function for motor-control move area
    def move_area(self):
        self.nanonis.ScanStop()
        self.nanonis.ScanFrameSet(0, 0, 5E-8, 5E-8)
        self.nanonis.ZCtrlWithdraw(1)
        time.sleep(0.5)
        self.nanonis.MotorMoveSet('Z-', 100)
        time.sleep(0.5)
        self.nanonis.MotorMoveSet('X-', 10)
        time.sleep(0.5)
        self.nanonis.MotorMoveSet('Y-', 10)
        # self.nanonis.ZCtrlOnOffSet(1)
        self.nanonis.AutoApproachOpen()
        time.sleep(0.5)
        self.nanonis.AutoApproachSet()
        print("area move succeed, wait for auto approach")
    # get Lock in XA
    def Lock_in_XA_get(self):
        self.lockin_window = LockInValueWindow(self.nanonis)
        self.lockin_window.show()
        self.lockin_window.update_value()
    def home_set(self):
        self.nanonis.ScanFrameSet(0, 0, 5E-8, 5E-8)
        self.nanonis.XYPosSet(0, 0)
    # window for model choose
    def show_choice_dialog(self):
        self.Integral_view.setValue(self.nanonis.ZCtrlGainGet()['I'] * 1E+9)
        self.proportional_view.setValue(self.nanonis.ZCtrlGainGet()['P'] * 1E+12)
        message_box = QMessageBox()
        message_box.setWindowTitle("Start Options")  # 设置窗口标题
        message_box.setIcon(QMessageBox.Question)  # 设置图标为疑问

        message_box.setText("Choose an option to start:")
        message_box.setInformativeText("Select 'New': the tip is initialized to the center and create a new log folder "
                                       "and create a new model.<br>"
                                       " Select 'Latest' : load the latest checkpoint.")

        new_button = message_box.addButton("New", QMessageBox.AcceptRole)
        new_button.setText("New")

        latest_button = message_box.addButton("Latest", QMessageBox.RejectRole)
        latest_button.setText("Latest")

        message_box.exec_()

        if message_box.clickedButton() == new_button:
            self.start_new_session()
            print("New session started.")
            self.queue_start(0)
            self.queue_mode('new')
        elif message_box.clickedButton() == latest_button:
            print("Continuing from the last session.")
            self.queue_start(0)
            self.queue_mode('latest')
    # start auto scan
    def start_new_session(self):
        size, okPressed = QInputDialog.getInt(self, "Input", "Enter the length of the image:", value=50)
        if okPressed and size > 0:
            count, okPressed = QInputDialog.getInt(self, "Input", "how many good images do you want:", value=100)
            self.nanonis.count_choose = count
            self.nanonis.Scan_edge = str(size) + 'n'
            if self.worker_thread is not None:
                self.worker_thread.quit()
                self.monitor_thread.stop_threads()
                self.worker_thread = None
                self.monitor_thread = None
                self.monitor_thread = monitor_thread(self.nanonis)
                self.worker_thread = NanonisThread(self.nanonis, self.shared_queues)
                self.monitor_thread.start_threads()
                self.nanonis.linescan_stop_event.clear()
                self.worker_thread.start()

            else:
                self.monitor_thread = monitor_thread(self.nanonis)
                self.worker_thread = NanonisThread(self.nanonis, self.shared_queues)
                self.monitor_thread.start_threads()
                self.worker_thread.start()

            print("New session started with length（n）:", size, "\n"
                  "Loop will stop while good images >", count)
        else:
            print("No dimensions provided or invalid input.")

    # def on_mode_changed(self, index):
    #     if index == 1:  # Manual 模式
    #         self.manualGroup.show()
    #         self.autoGroup.hide()
    #     else:  # Auto 模式
    #         self.manualGroup.hide()
    #         self.autoGroup.show()
    # a queue for pass mode
    def queue_mode(self, mode):
        queue_mode = self.shared_queues[1]
        queue_mode.put(mode)
        return queue_mode
    # start signal
    def queue_start(self, n):
        queue_start = self.shared_queues[0]
        queue_start.put(n)
        return queue_start
    # tip sharp thread queue
    def queue_tipsharp(self, n):
        tip_lift = str(n) + 'n'
        self.nanonis.TipShaper_set(tip_lift)
        print(f'Tip shaper occurred with {tip_lift}')
        self.show_move_choice()
    # scan speed set
    def on_apply_button_clicked(self):
        if self.nanonis.line_scan_signal == 0:
            time_per_frame = self.time.value()
            self.nanonis.speed_set(time_per_frame)
    # control scan state
    def scan_control(self, n):
        self.nanonis.controls_set(n)

    def queue_setpoint(self):
        setpoint_value = self.setpoint_value.value() / 1E+12
        self.nanonis.SetpointSet(setpoint_value)
        print(f'setpoint set with {self.setpoint_value.value()}pA')

    def on_set_clicked(self):
        proportional_value = self.proportional_view.value() / 1E+12
        integral_value = self.Integral_view.value() / 1E+9
        self.nanonis.Gainset(proportional_value, integral_value)
    # open folder
    def selectFolder(self):
        current_directory = os.getcwd()
        folder_path = os.path.join(current_directory, "./log")
        os.startfile(folder_path)
    # a window to ask if move tip
    def show_move_choice(self):
        message_box = QMessageBox()
        message_box.setWindowTitle("Start Options")
        message_box.setIcon(QMessageBox.Question)

        message_box.setText("Choose an option to start:")
        message_box.setInformativeText("Select 'Move': move to next scan point.<br>"
                                       " Select 'Continue' : continue scan this scan point.")

        move_button = message_box.addButton("Move", QMessageBox.AcceptRole)
        move_button.setText("Move")

        continue_button = message_box.addButton("Continue", QMessageBox.RejectRole)
        continue_button.setText("Continue")

        message_box.exec_()

        if message_box.clickedButton() == move_button:
            self.nanonis.Tipshaper_signal = 1
            print("move to next scan point.")
        elif message_box.clickedButton() == continue_button:
            print("Continuing scan.")
    # pulse button
    def pulse(self):
        bias_value = self.BiasValue.value()
        width_value = self.WidthValue.value() / 1000
        self.nanonis.bias_pulse_set(width_value, bias_value)

from PyQt5.QtCore import QThread, Qt, QTimer, pyqtSignal
from queue import Empty
# show lock in window
class LockInValueWindow(QWidget):
    def __init__(self, nanonis_instance, parent=None):
        super().__init__(parent)
        self.nanonis = nanonis_instance
        self.value_label = QLabel("LockIn_X(A): ", self)
        self.value_display = QLabel("Loading...", self)
        layout = QVBoxLayout()
        layout.addWidget(self.value_label)
        layout.addWidget(self.value_display)
        self.setLayout(layout)
        self.setFixedSize(300, 100)
        self.setWindowTitle("电容")
        self.value = "0"  # 初始值
        self.timer = QTimer(self)  # 创建定时器
        self.timer.timeout.connect(self.update_value)
        self.timer.start(100)
    def update_value(self):
        try:
            value = self.nanonis.SignalValsGet(86)['0']
            self.value_display.setText(f"{value}")
        except Exception as e:
            print(f"update :{e}")

# thread for scan loop
class NanonisThread(QThread):
    def __init__(self, nanonis_instance, shared_queues, parent=None):
        super(NanonisThread, self).__init__(parent)
        self.controls_queue = shared_queues[0]
        self.nanonis = nanonis_instance
        self.mode = shared_queues[1]

    def run_scanning_loop(self, mode):
        self.nanonis.mode = mode
        self.nanonis.tip_init(
            mode=mode)  # deflaut mode is 'new' mode = 'new' : the tip is initialized to the center and create a new log folder, mode = 'latest' : load the latest checkpoint

        self.nanonis.DQN_init(
            mode=mode)  # deflaut mode is 'latest' mode = 'new' : create a new model, mode = 'latest' : load the latest checkpoint

        # self.nanonis.monitor_thread_activate()  # activate the monitor thread
        self.nanonis.ScanSpeedSet(0.1, 0.1, 2E-1, 2E-1, 2, 1)
        self.nanonis.ScanBufferSet(208, 208, self.nanonis.signal_channel_list)
        while self.nanonis.good_image_count <= self.nanonis.count_choose:
            while tip_in_boundary(self.nanonis.inter_closest, self.nanonis.plane_size, self.nanonis.real_scan_factor):

                self.nanonis.move_to_next_point()  # move the scan area to the next point

                self.nanonis.AdjustTip_flag = self.nanonis.AdjustTipToPiezoCenter()  # check & adjust the tip to the center of the piezo

                self.nanonis.line_scan_thread_activate()  # activate the line scan, producer-consumer architecture, pre-check the tip and sample

                self.nanonis.batch_scan_producer( self.nanonis.nanocoodinate, self.nanonis.Scan_edge,
                                                 self.nanonis.scan_square_Buffer_pix,
                                                 0, )  # Scan the area

                scan_qulity = self.nanonis.image_recognition()  # assement the scan qulity

                self.nanonis.create_trajectory(scan_qulity)  # create the trajectory

                if scan_qulity == 0 and self.nanonis.agent_upgrate:
                    self.nanonis.DQN_upgrate()  # optimize the model and update the target network

                self.nanonis.save_checkpoint()  # save the checkpoint

            self.nanonis.move_to_next_area()

        self.nanonis.Withdraw()
    def run(self):
        while True:
            try:
                controls = self.controls_queue.get()
                mode = self.mode.get()
                if controls is None:
                    break
                if controls == 0:
                    self.run_scanning_loop(mode)
            except Empty:
                continue
            except Exception as e:
                break

# pass image information
class imageThread(QThread):
    def __init__(self, nanonis_instance, ui_instance, parent=None):
        super(imageThread, self).__init__(parent)
        self.nanonis = nanonis_instance
        self.image = None
        self.ui = ui_instance

    def run(self):
        while True:
            try:
                if not self.nanonis.ScandataQueue_1.empty():
                    self.image = self.nanonis.ScandataQueue_1.get()
                    Scan_data_for = self.image['Scan_data_for']['data']
                    Scan_data_back = self.image['Scan_data_back']['data']
                    # preprocess the scan data, and save the scan data and image
                    image_for = linear_normalize_whole(Scan_data_for)
                    image_back = linear_normalize_whole(Scan_data_back)
                    self.nanonis.scale_image_for = image_for
                    self.nanonis.scale_image_back = image_back
                    image_for = np.array(image_for, dtype=np.uint8)
                    image_back = np.array(image_back, dtype=np.uint8)
                    self.nanonis.scale_signal = 1
                    # 创建 QImage 对象
                    qt_image_for = QImage(
                        image_for.data,
                        image_for.shape[1],  # 宽度
                        image_for.shape[0],  # 高度
                        image_for.strides[0],  # 每行的字节数
                        QImage.Format_Grayscale8  # 使用 QImage 的类属性而不是 QtGui 的
                    )
                    qt_image_back = QImage(
                        image_back.data,
                        image_back.shape[1],  # 宽度
                        image_back.shape[0],  # 高度
                        image_back.strides[0],  # 每行的字节数
                        QImage.Format_Grayscale8  # 使用 QImage 的类属性而不是 QtGui 的
                    )
                    # 将 QImage 转换为 QPixmap
                    pixmap_for = QPixmap.fromImage(qt_image_for)
                    pixmap_back = QPixmap.fromImage(qt_image_back)
                    scaled_pixmap_for = pixmap_for.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    scaled_pixmap_back = pixmap_back.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    # 更新图像显示
                    self.ui.image_for.setPixmap(scaled_pixmap_for)
                    self.ui.image_for.show()
                    self.ui.image_back.setPixmap(scaled_pixmap_back)
                    self.ui.image_back.show()
            except Empty:  # 队列为空时捕获 Empty 异常
                time.sleep(1)
                continue
            except Exception as e:  # 其他异常处理
                print(f"An error occurred in imagethread: {e}")
                break  # 退出循环

# pass tip path information
class tippathThread(QThread):
    def __init__(self, nanonis_instance, ui_instance, parent=None):
        super(tippathThread, self).__init__(parent)
        self.nanonis = nanonis_instance
        self.image = None
        self.ui = ui_instance

    def run(self):
        while True:
            try:
                if not self.nanonis.tipdataQueue.empty():
                    self.image = self.nanonis.tipdataQueue.get()
                    # preprocess the scan data, and save the scan data and image
                    image_for = np.array(self.image, dtype=np.uint8)
                    # 创建 QImage 对象
                    qt_image_for = QImage(
                        image_for.data,
                        image_for.shape[1],  # 宽度
                        image_for.shape[0],  # 高度
                        image_for.strides[0],  # 每行的字节数
                        QImage.Format_BGR888  # 使用 QImage 的类属性而不是 QtGui 的
                    )
                    # 将 QImage 转换为 QPixmap
                    pixmap_for = QPixmap.fromImage(qt_image_for)
                    scaled_pixmap_for = pixmap_for.scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    # 更新图像显示
                    self.ui.image_tip_path.setPixmap(scaled_pixmap_for)
                    self.ui.image_tip_path.show()

            except Empty:  # 队列为空时捕获 Empty 异常
                time.sleep(1)
                continue
            except Exception as e:  # 其他异常处理
                print(f"An error occurred in tippathThread: {e}")
                break  # 退出循环
# thread to fix tip
class fixtipThread(QThread):
    def __init__(self, nanonis_instance,shared_queues, parent=None):
        super(fixtipThread, self).__init__(parent)
        self.nanonis = nanonis_instance
        self.signal = shared_queues[2]
    def run(self):
        while True:
            signal = self.signal.get()
            if signal == 1:
                while True:
                    self.nanonis.bias_pulse()
                    time.sleep(3)
                    try:
                        signal = self.signal.get_nowait()
                    except Empty:
                        continue
                    if signal == 0:
                        break
# other monitor thread
class monitor_thread():
    def __init__(self, nanonis_instance):
        self.nanonis = nanonis_instance
        self.Safe_Tip_thread = threading.Thread(target=self.nanonis.SafeTipthreading, args=('5n', 100), daemon=True)
        self.tip_visualization_thread = threading.Thread(target=self.nanonis.tip_path_visualization, daemon=True)
        self.batch_scan_consumer_thread = threading.Thread(target=self.nanonis.batch_scan_consumer, daemon=True)

    def start_threads(self):
        self.Safe_Tip_thread.start()
        self.batch_scan_consumer_thread.start()
        self.tip_visualization_thread.start()
    def stop_threads(self):
        # 设置停止事件
        self.nanonis.batch_stop_event.set()
        self.nanonis.savetip_stop_event.set()
        self.nanonis.linescan_stop_event.set()
        self.nanonis.tippath_stop_event.set()
        # 等待线程结束
        self.Safe_Tip_thread.join()
        self.tip_visualization_thread.join()
        self.batch_scan_consumer_thread.join()
        # 重置停止事件
        self.nanonis.tippath_stop_event.clear()
        self.nanonis.batch_stop_event.clear()
        self.nanonis.savetip_stop_event.clear()
