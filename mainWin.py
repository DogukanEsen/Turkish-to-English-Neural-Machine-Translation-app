# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainWin.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(775, 712)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(-10, 0, 771, 681))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.frame = QtWidgets.QFrame(self.tab)
        self.frame.setGeometry(QtCore.QRect(0, 0, 781, 661))
        self.frame.setStyleSheet("background-color: rgb(251, 249, 255);\n"
"")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setMinimumSize(QtCore.QSize(100, 125))
        self.frame_2.setStyleSheet("background-color: rgb(246, 240, 255);")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.frame_5 = QtWidgets.QFrame(self.frame_2)
        self.frame_5.setGeometry(QtCore.QRect(110, 0, 501, 131))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_5)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label_4 = QtWidgets.QLabel(self.frame_5)
        self.label_4.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap("icons/reshot-icon-united-kingdom-2GKF39DHUP.svg"))
        self.label_4.setScaledContents(True)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout.addWidget(self.label_4)
        self.label_3 = QtWidgets.QLabel(self.frame_5)
        self.label_3.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.frame_6 = QtWidgets.QFrame(self.frame_5)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.gridLayout = QtWidgets.QGridLayout(self.frame_6)
        self.gridLayout.setObjectName("gridLayout")
        self.frame_7 = QtWidgets.QFrame(self.frame_6)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.gridLayout.addWidget(self.frame_7, 0, 0, 1, 1)
        self.frame_8 = QtWidgets.QFrame(self.frame_6)
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.gridLayout.addWidget(self.frame_8, 2, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.frame_6)
        self.label_5.setStyleSheet("font: 36pt \"MS Shell Dlg 2\";")
        self.label_5.setText("")
        self.label_5.setPixmap(QtGui.QPixmap("icons/reshot-icon-right-arrow-UCA8NGYZDJ.svg"))
        self.label_5.setScaledContents(True)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 1, 0, 1, 2)
        self.horizontalLayout.addWidget(self.frame_6)
        self.label_2 = QtWidgets.QLabel(self.frame_5)
        self.label_2.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.label = QtWidgets.QLabel(self.frame_5)
        self.label.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("icons/reshot-icon-turkey-XBVCFG6SMY.svg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.verticalLayout.addWidget(self.frame_2, 0, QtCore.Qt.AlignTop)
        self.frame_4 = QtWidgets.QFrame(self.frame)
        self.frame_4.setMinimumSize(QtCore.QSize(0, 250))
        self.frame_4.setMaximumSize(QtCore.QSize(16777215, 250))
        self.frame_4.setStyleSheet("background-color: rgb(246, 240, 255);")
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.frame_9 = QtWidgets.QFrame(self.frame_4)
        self.frame_9.setGeometry(QtCore.QRect(20, 230, 120, 80))
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.label_6 = QtWidgets.QLabel(self.frame_4)
        self.label_6.setGeometry(QtCore.QRect(40, 20, 101, 21))
        self.label_6.setStyleSheet("font: 11pt \"Arial\";")
        self.label_6.setObjectName("label_6")
        self.engText = QtWidgets.QPlainTextEdit(self.frame_4)
        self.engText.setGeometry(QtCore.QRect(40, 50, 691, 181))
        self.engText.setObjectName("engText")
        self.translateBtn = QtWidgets.QPushButton(self.frame_4)
        self.translateBtn.setGeometry(QtCore.QRect(640, 10, 93, 28))
        self.translateBtn.setStyleSheet("background-color: rgb(255, 170, 0);\n"
"selection-background-color: rgb(255, 85, 0);")
        self.translateBtn.setObjectName("translateBtn")
        self.verticalLayout.addWidget(self.frame_4)
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setMinimumSize(QtCore.QSize(0, 250))
        self.frame_3.setMaximumSize(QtCore.QSize(16777215, 250))
        self.frame_3.setStyleSheet("background-color: rgb(246, 240, 255);")
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.trText = QtWidgets.QPlainTextEdit(self.frame_3)
        self.trText.setEnabled(False)
        self.trText.setGeometry(QtCore.QRect(40, 50, 691, 181))
        self.trText.setObjectName("trText")
        self.label_7 = QtWidgets.QLabel(self.frame_3)
        self.label_7.setGeometry(QtCore.QRect(40, 20, 101, 21))
        self.label_7.setStyleSheet("font: 11pt \"Arial\";")
        self.label_7.setObjectName("label_7")
        self.verticalLayout.addWidget(self.frame_3, 0, QtCore.Qt.AlignBottom)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.frame_10 = QtWidgets.QFrame(self.tab_2)
        self.frame_10.setGeometry(QtCore.QRect(0, 0, 781, 661))
        self.frame_10.setStyleSheet("background-color: rgb(251, 249, 255);\n"
"")
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_10)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame_11 = QtWidgets.QFrame(self.frame_10)
        self.frame_11.setMinimumSize(QtCore.QSize(100, 125))
        self.frame_11.setStyleSheet("background-color: rgb(246, 240, 255);")
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.frame_12 = QtWidgets.QFrame(self.frame_11)
        self.frame_12.setGeometry(QtCore.QRect(110, 0, 501, 131))
        self.frame_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_12)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem1 = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.label_8 = QtWidgets.QLabel(self.frame_12)
        self.label_8.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label_8.setText("")
        self.label_8.setPixmap(QtGui.QPixmap("icons/reshot-icon-united-kingdom-2GKF39DHUP.svg"))
        self.label_8.setScaledContents(True)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_2.addWidget(self.label_8)
        self.label_9 = QtWidgets.QLabel(self.frame_12)
        self.label_9.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_2.addWidget(self.label_9)
        self.frame_13 = QtWidgets.QFrame(self.frame_12)
        self.frame_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_13.setObjectName("frame_13")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_13)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.frame_14 = QtWidgets.QFrame(self.frame_13)
        self.frame_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_14.setObjectName("frame_14")
        self.gridLayout_2.addWidget(self.frame_14, 0, 0, 1, 1)
        self.frame_15 = QtWidgets.QFrame(self.frame_13)
        self.frame_15.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_15.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_15.setObjectName("frame_15")
        self.gridLayout_2.addWidget(self.frame_15, 2, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.frame_13)
        self.label_10.setStyleSheet("font: 36pt \"MS Shell Dlg 2\";")
        self.label_10.setText("")
        self.label_10.setPixmap(QtGui.QPixmap("icons/reshot-icon-right-arrow-UCA8NGYZDJ.svg"))
        self.label_10.setScaledContents(True)
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 1, 0, 1, 2)
        self.horizontalLayout_2.addWidget(self.frame_13)
        self.label_11 = QtWidgets.QLabel(self.frame_12)
        self.label_11.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_2.addWidget(self.label_11)
        self.label_12 = QtWidgets.QLabel(self.frame_12)
        self.label_12.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label_12.setText("")
        self.label_12.setPixmap(QtGui.QPixmap("icons/reshot-icon-turkey-XBVCFG6SMY.svg"))
        self.label_12.setScaledContents(True)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_2.addWidget(self.label_12)
        self.verticalLayout_2.addWidget(self.frame_11, 0, QtCore.Qt.AlignTop)
        self.frame_16 = QtWidgets.QFrame(self.frame_10)
        self.frame_16.setMinimumSize(QtCore.QSize(0, 250))
        self.frame_16.setMaximumSize(QtCore.QSize(16777215, 250))
        self.frame_16.setStyleSheet("background-color: rgb(246, 240, 255);")
        self.frame_16.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_16.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_16.setObjectName("frame_16")
        self.frame_17 = QtWidgets.QFrame(self.frame_16)
        self.frame_17.setGeometry(QtCore.QRect(20, 230, 120, 80))
        self.frame_17.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_17.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_17.setObjectName("frame_17")
        self.label_13 = QtWidgets.QLabel(self.frame_16)
        self.label_13.setGeometry(QtCore.QRect(40, 20, 101, 21))
        self.label_13.setStyleSheet("font: 11pt \"Arial\";")
        self.label_13.setObjectName("label_13")
        self.engText_2 = QtWidgets.QPlainTextEdit(self.frame_16)
        self.engText_2.setGeometry(QtCore.QRect(40, 50, 691, 181))
        self.engText_2.setObjectName("engText_2")
        self.translateBtn_2 = QtWidgets.QPushButton(self.frame_16)
        self.translateBtn_2.setGeometry(QtCore.QRect(640, 10, 93, 28))
        self.translateBtn_2.setStyleSheet("background-color: rgb(255, 170, 0);\n"
"selection-background-color: rgb(255, 85, 0);")
        self.translateBtn_2.setObjectName("translateBtn_2")
        self.label_14 = QtWidgets.QLabel(self.frame_16)
        self.label_14.setGeometry(QtCore.QRect(330, 10, 301, 31))
        self.label_14.setStyleSheet("font: 11pt \"Arial\";")
        self.label_14.setObjectName("label_14")
        self.verticalLayout_2.addWidget(self.frame_16)
        self.frame_18 = QtWidgets.QFrame(self.frame_10)
        self.frame_18.setMinimumSize(QtCore.QSize(0, 250))
        self.frame_18.setMaximumSize(QtCore.QSize(16777215, 250))
        self.frame_18.setStyleSheet("background-color: rgb(246, 240, 255);")
        self.frame_18.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_18.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_18.setObjectName("frame_18")
        self.trText_2 = QtWidgets.QPlainTextEdit(self.frame_18)
        self.trText_2.setEnabled(False)
        self.trText_2.setGeometry(QtCore.QRect(40, 50, 691, 181))
        self.trText_2.setObjectName("trText_2")
        self.label_15 = QtWidgets.QLabel(self.frame_18)
        self.label_15.setGeometry(QtCore.QRect(40, 20, 101, 21))
        self.label_15.setStyleSheet("font: 11pt \"Arial\";")
        self.label_15.setObjectName("label_15")
        self.verticalLayout_2.addWidget(self.frame_18, 0, QtCore.Qt.AlignBottom)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.frame_19 = QtWidgets.QFrame(self.tab_5)
        self.frame_19.setGeometry(QtCore.QRect(0, 0, 781, 661))
        self.frame_19.setStyleSheet("background-color: rgb(251, 249, 255);\n"
"")
        self.frame_19.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_19.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_19.setObjectName("frame_19")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_19)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.frame_28 = QtWidgets.QFrame(self.frame_19)
        self.frame_28.setMinimumSize(QtCore.QSize(100, 125))
        self.frame_28.setStyleSheet("background-color: rgb(246, 240, 255);")
        self.frame_28.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_28.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_28.setObjectName("frame_28")
        self.frame_29 = QtWidgets.QFrame(self.frame_28)
        self.frame_29.setGeometry(QtCore.QRect(110, 0, 501, 131))
        self.frame_29.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_29.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_29.setObjectName("frame_29")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_29)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem2 = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem2)
        self.label_24 = QtWidgets.QLabel(self.frame_29)
        self.label_24.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label_24.setText("")
        self.label_24.setPixmap(QtGui.QPixmap("icons/reshot-icon-united-kingdom-2GKF39DHUP.svg"))
        self.label_24.setScaledContents(True)
        self.label_24.setObjectName("label_24")
        self.horizontalLayout_4.addWidget(self.label_24)
        self.label_25 = QtWidgets.QLabel(self.frame_29)
        self.label_25.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label_25.setObjectName("label_25")
        self.horizontalLayout_4.addWidget(self.label_25)
        self.frame_30 = QtWidgets.QFrame(self.frame_29)
        self.frame_30.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_30.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_30.setObjectName("frame_30")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_30)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.frame_31 = QtWidgets.QFrame(self.frame_30)
        self.frame_31.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_31.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_31.setObjectName("frame_31")
        self.gridLayout_4.addWidget(self.frame_31, 0, 0, 1, 1)
        self.frame_32 = QtWidgets.QFrame(self.frame_30)
        self.frame_32.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_32.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_32.setObjectName("frame_32")
        self.gridLayout_4.addWidget(self.frame_32, 2, 0, 1, 1)
        self.label_26 = QtWidgets.QLabel(self.frame_30)
        self.label_26.setStyleSheet("font: 36pt \"MS Shell Dlg 2\";")
        self.label_26.setText("")
        self.label_26.setPixmap(QtGui.QPixmap("icons/reshot-icon-right-arrow-UCA8NGYZDJ.svg"))
        self.label_26.setScaledContents(True)
        self.label_26.setObjectName("label_26")
        self.gridLayout_4.addWidget(self.label_26, 1, 0, 1, 2)
        self.horizontalLayout_4.addWidget(self.frame_30)
        self.label_27 = QtWidgets.QLabel(self.frame_29)
        self.label_27.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label_27.setObjectName("label_27")
        self.horizontalLayout_4.addWidget(self.label_27)
        self.label_28 = QtWidgets.QLabel(self.frame_29)
        self.label_28.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label_28.setText("")
        self.label_28.setPixmap(QtGui.QPixmap("icons/reshot-icon-turkey-XBVCFG6SMY.svg"))
        self.label_28.setScaledContents(True)
        self.label_28.setObjectName("label_28")
        self.horizontalLayout_4.addWidget(self.label_28)
        self.verticalLayout_4.addWidget(self.frame_28, 0, QtCore.Qt.AlignTop)
        self.frame_33 = QtWidgets.QFrame(self.frame_19)
        self.frame_33.setMinimumSize(QtCore.QSize(0, 250))
        self.frame_33.setMaximumSize(QtCore.QSize(16777215, 250))
        self.frame_33.setStyleSheet("background-color: rgb(246, 240, 255);")
        self.frame_33.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_33.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_33.setObjectName("frame_33")
        self.frame_34 = QtWidgets.QFrame(self.frame_33)
        self.frame_34.setGeometry(QtCore.QRect(20, 230, 120, 80))
        self.frame_34.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_34.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_34.setObjectName("frame_34")
        self.label_29 = QtWidgets.QLabel(self.frame_33)
        self.label_29.setGeometry(QtCore.QRect(40, 20, 101, 21))
        self.label_29.setStyleSheet("font: 11pt \"Arial\";")
        self.label_29.setObjectName("label_29")
        self.engText_4 = QtWidgets.QPlainTextEdit(self.frame_33)
        self.engText_4.setGeometry(QtCore.QRect(40, 50, 691, 181))
        self.engText_4.setObjectName("engText_4")
        self.translateBtn_4 = QtWidgets.QPushButton(self.frame_33)
        self.translateBtn_4.setGeometry(QtCore.QRect(640, 10, 93, 28))
        self.translateBtn_4.setStyleSheet("background-color: rgb(255, 170, 0);\n"
"selection-background-color: rgb(255, 85, 0);")
        ###
        self.translateBtn_4.setObjectName("translateBtn_4")
        self.label_30 = QtWidgets.QLabel(self.frame_33)
        self.label_30.setGeometry(QtCore.QRect(340, 10, 291, 31))
        self.label_30.setStyleSheet("font: 11pt \"Arial\";")
        self.label_30.setObjectName("label_30")
        self.verticalLayout_4.addWidget(self.frame_33)
        self.frame_35 = QtWidgets.QFrame(self.frame_19)
        self.frame_35.setMinimumSize(QtCore.QSize(0, 250))
        self.frame_35.setMaximumSize(QtCore.QSize(16777215, 250))
        self.frame_35.setStyleSheet("background-color: rgb(246, 240, 255);")
        self.frame_35.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_35.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_35.setObjectName("frame_35")
        self.trText_4 = QtWidgets.QPlainTextEdit(self.frame_35)
        self.trText_4.setEnabled(False)
        self.trText_4.setGeometry(QtCore.QRect(40, 50, 691, 181))
        self.trText_4.setObjectName("trText_4")
        self.label_31 = QtWidgets.QLabel(self.frame_35)
        self.label_31.setGeometry(QtCore.QRect(40, 20, 101, 21))
        self.label_31.setStyleSheet("font: 11pt \"Arial\";")
        self.label_31.setObjectName("label_31")
        self.verticalLayout_4.addWidget(self.frame_35, 0, QtCore.Qt.AlignBottom)
        self.tabWidget.addTab(self.tab_5, "")
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")
        self.frame_36 = QtWidgets.QFrame(self.tab_6)
        self.frame_36.setGeometry(QtCore.QRect(0, 0, 781, 661))
        self.frame_36.setStyleSheet("background-color: rgb(251, 249, 255);\n"
"")
        self.frame_36.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_36.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_36.setObjectName("frame_36")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_36)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.frame_37 = QtWidgets.QFrame(self.frame_36)
        self.frame_37.setMinimumSize(QtCore.QSize(100, 125))
        self.frame_37.setStyleSheet("background-color: rgb(246, 240, 255);")
        self.frame_37.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_37.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_37.setObjectName("frame_37")
        self.frame_38 = QtWidgets.QFrame(self.frame_37)
        self.frame_38.setGeometry(QtCore.QRect(110, 0, 501, 131))
        self.frame_38.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_38.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_38.setObjectName("frame_38")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame_38)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem3 = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem3)
        self.label_32 = QtWidgets.QLabel(self.frame_38)
        self.label_32.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label_32.setText("")
        self.label_32.setPixmap(QtGui.QPixmap("icons/reshot-icon-turkey-XBVCFG6SMY.svg"))
        self.label_32.setScaledContents(True)
        self.label_32.setObjectName("label_32")
        self.horizontalLayout_5.addWidget(self.label_32)
        self.label_33 = QtWidgets.QLabel(self.frame_38)
        self.label_33.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label_33.setObjectName("label_33")
        self.horizontalLayout_5.addWidget(self.label_33)
        self.frame_39 = QtWidgets.QFrame(self.frame_38)
        self.frame_39.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_39.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_39.setObjectName("frame_39")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.frame_39)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.frame_40 = QtWidgets.QFrame(self.frame_39)
        self.frame_40.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_40.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_40.setObjectName("frame_40")
        self.gridLayout_5.addWidget(self.frame_40, 0, 0, 1, 1)
        self.frame_41 = QtWidgets.QFrame(self.frame_39)
        self.frame_41.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_41.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_41.setObjectName("frame_41")
        self.gridLayout_5.addWidget(self.frame_41, 2, 0, 1, 1)
        self.label_34 = QtWidgets.QLabel(self.frame_39)
        self.label_34.setStyleSheet("font: 36pt \"MS Shell Dlg 2\";")
        self.label_34.setText("")
        self.label_34.setPixmap(QtGui.QPixmap("icons/reshot-icon-right-arrow-UCA8NGYZDJ.svg"))
        self.label_34.setScaledContents(True)
        self.label_34.setObjectName("label_34")
        self.gridLayout_5.addWidget(self.label_34, 1, 0, 1, 2)
        self.horizontalLayout_5.addWidget(self.frame_39)
        self.label_35 = QtWidgets.QLabel(self.frame_38)
        self.label_35.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label_35.setObjectName("label_35")
        self.horizontalLayout_5.addWidget(self.label_35)
        self.label_36 = QtWidgets.QLabel(self.frame_38)
        self.label_36.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label_36.setText("")
        self.label_36.setPixmap(QtGui.QPixmap("icons/reshot-icon-united-kingdom-2GKF39DHUP.svg"))
        self.label_36.setScaledContents(True)
        self.label_36.setObjectName("label_36")
        self.horizontalLayout_5.addWidget(self.label_36)
        self.verticalLayout_5.addWidget(self.frame_37, 0, QtCore.Qt.AlignTop)
        self.frame_42 = QtWidgets.QFrame(self.frame_36)
        self.frame_42.setMinimumSize(QtCore.QSize(0, 250))
        self.frame_42.setMaximumSize(QtCore.QSize(16777215, 250))
        self.frame_42.setStyleSheet("background-color: rgb(246, 240, 255);")
        self.frame_42.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_42.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_42.setObjectName("frame_42")
        self.frame_43 = QtWidgets.QFrame(self.frame_42)
        self.frame_43.setGeometry(QtCore.QRect(20, 230, 120, 80))
        self.frame_43.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_43.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_43.setObjectName("frame_43")
        self.label_37 = QtWidgets.QLabel(self.frame_42)
        self.label_37.setGeometry(QtCore.QRect(40, 20, 101, 21))
        self.label_37.setStyleSheet("font: 11pt \"Arial\";")
        self.label_37.setObjectName("label_37")
        self.engText_5 = QtWidgets.QPlainTextEdit(self.frame_42)
        self.engText_5.setGeometry(QtCore.QRect(40, 50, 691, 181))
        self.engText_5.setObjectName("engText_5")
        self.translateBtn_5 = QtWidgets.QPushButton(self.frame_42)
        self.translateBtn_5.setGeometry(QtCore.QRect(640, 10, 93, 28))
        self.translateBtn_5.setStyleSheet("background-color: rgb(255, 170, 0);\n"
"selection-background-color: rgb(255, 85, 0);")
        self.translateBtn_5.setObjectName("translateBtn_5")
        ###
        self.label_38 = QtWidgets.QLabel(self.frame_42)
        self.label_38.setGeometry(QtCore.QRect(380, 10, 251, 31))
        self.label_38.setStyleSheet("font: 11pt \"Arial\";")
        self.label_38.setObjectName("label_38")
        self.verticalLayout_5.addWidget(self.frame_42)
        self.frame_44 = QtWidgets.QFrame(self.frame_36)
        self.frame_44.setMinimumSize(QtCore.QSize(0, 250))
        self.frame_44.setMaximumSize(QtCore.QSize(16777215, 250))
        self.frame_44.setStyleSheet("background-color: rgb(246, 240, 255);")
        self.frame_44.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_44.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_44.setObjectName("frame_44")
        self.trText_5 = QtWidgets.QPlainTextEdit(self.frame_44)
        self.trText_5.setEnabled(False)
        self.trText_5.setGeometry(QtCore.QRect(40, 50, 691, 181))
        self.trText_5.setObjectName("trText_5")
        self.label_39 = QtWidgets.QLabel(self.frame_44)
        self.label_39.setGeometry(QtCore.QRect(40, 20, 101, 21))
        self.label_39.setStyleSheet("font: 11pt \"Arial\";")
        self.label_39.setObjectName("label_39")
        self.verticalLayout_5.addWidget(self.frame_44, 0, QtCore.Qt.AlignBottom)
        self.tabWidget.addTab(self.tab_6, "")
        self.tab_7 = QtWidgets.QWidget()
        self.tab_7.setObjectName("tab_7")
        self.frame_45 = QtWidgets.QFrame(self.tab_7)
        self.frame_45.setGeometry(QtCore.QRect(0, 0, 781, 661))
        self.frame_45.setStyleSheet("background-color: rgb(251, 249, 255);\n"
"")
        self.frame_45.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_45.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_45.setObjectName("frame_45")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_45)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.frame_46 = QtWidgets.QFrame(self.frame_45)
        self.frame_46.setMinimumSize(QtCore.QSize(100, 125))
        self.frame_46.setStyleSheet("background-color: rgb(246, 240, 255);")
        self.frame_46.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_46.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_46.setObjectName("frame_46")
        self.frame_47 = QtWidgets.QFrame(self.frame_46)
        self.frame_47.setGeometry(QtCore.QRect(90, 0, 581, 131))
        self.frame_47.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_47.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_47.setObjectName("frame_47")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.frame_47)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        spacerItem4 = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem4)
        self.label_40 = QtWidgets.QLabel(self.frame_47)
        self.label_40.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label_40.setText("")
        self.label_40.setPixmap(QtGui.QPixmap("icons/reshot-icon-turkey-XBVCFG6SMY.svg"))
        self.label_40.setScaledContents(True)
        self.label_40.setObjectName("label_40")
        self.horizontalLayout_6.addWidget(self.label_40)
        self.label_41 = QtWidgets.QLabel(self.frame_47)
        self.label_41.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label_41.setObjectName("label_41")
        self.horizontalLayout_6.addWidget(self.label_41)
        self.frame_48 = QtWidgets.QFrame(self.frame_47)
        self.frame_48.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_48.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_48.setObjectName("frame_48")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.frame_48)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.frame_49 = QtWidgets.QFrame(self.frame_48)
        self.frame_49.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_49.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_49.setObjectName("frame_49")
        self.gridLayout_6.addWidget(self.frame_49, 0, 0, 1, 1)
        self.frame_50 = QtWidgets.QFrame(self.frame_48)
        self.frame_50.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_50.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_50.setObjectName("frame_50")
        self.gridLayout_6.addWidget(self.frame_50, 2, 0, 1, 1)
        self.label_42 = QtWidgets.QLabel(self.frame_48)
        self.label_42.setStyleSheet("font: 36pt \"MS Shell Dlg 2\";")
        self.label_42.setText("")
        self.label_42.setPixmap(QtGui.QPixmap("icons/reshot-icon-right-arrow-UCA8NGYZDJ.svg"))
        self.label_42.setScaledContents(True)
        self.label_42.setObjectName("label_42")
        self.gridLayout_6.addWidget(self.label_42, 1, 0, 1, 2)
        self.horizontalLayout_6.addWidget(self.frame_48)
        self.label_43 = QtWidgets.QLabel(self.frame_47)
        self.label_43.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label_43.setObjectName("label_43")
        self.horizontalLayout_6.addWidget(self.label_43)
        self.label_44 = QtWidgets.QLabel(self.frame_47)
        self.label_44.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label_44.setText("")
        self.label_44.setPixmap(QtGui.QPixmap("icons/reshot-icon-spain-ZXJ9R2SENK.svg"))
        self.label_44.setScaledContents(True)
        self.label_44.setObjectName("label_44")
        self.horizontalLayout_6.addWidget(self.label_44)
        self.verticalLayout_6.addWidget(self.frame_46, 0, QtCore.Qt.AlignTop)
        self.frame_51 = QtWidgets.QFrame(self.frame_45)
        self.frame_51.setMinimumSize(QtCore.QSize(0, 250))
        self.frame_51.setMaximumSize(QtCore.QSize(16777215, 250))
        self.frame_51.setStyleSheet("background-color: rgb(246, 240, 255);")
        self.frame_51.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_51.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_51.setObjectName("frame_51")
        self.frame_52 = QtWidgets.QFrame(self.frame_51)
        self.frame_52.setGeometry(QtCore.QRect(20, 230, 120, 80))
        self.frame_52.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_52.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_52.setObjectName("frame_52")
        self.label_45 = QtWidgets.QLabel(self.frame_51)
        self.label_45.setGeometry(QtCore.QRect(40, 20, 101, 21))
        self.label_45.setStyleSheet("font: 11pt \"Arial\";")
        self.label_45.setObjectName("label_45")
        self.engText_6 = QtWidgets.QPlainTextEdit(self.frame_51)
        self.engText_6.setGeometry(QtCore.QRect(40, 50, 691, 181))
        self.engText_6.setObjectName("engText_6")
        self.translateBtn_6 = QtWidgets.QPushButton(self.frame_51)
        self.translateBtn_6.setGeometry(QtCore.QRect(640, 10, 93, 28))
        self.translateBtn_6.setStyleSheet("background-color: rgb(255, 170, 0);\n"
"selection-background-color: rgb(255, 85, 0);")
        self.translateBtn_6.setObjectName("translateBtn_6")
        ###
        self.label_46 = QtWidgets.QLabel(self.frame_51)
        self.label_46.setGeometry(QtCore.QRect(380, 10, 251, 31))
        self.label_46.setStyleSheet("font: 11pt \"Arial\";")
        self.label_46.setObjectName("label_46")
        self.verticalLayout_6.addWidget(self.frame_51)
        self.frame_53 = QtWidgets.QFrame(self.frame_45)
        self.frame_53.setMinimumSize(QtCore.QSize(0, 250))
        self.frame_53.setMaximumSize(QtCore.QSize(16777215, 250))
        self.frame_53.setStyleSheet("background-color: rgb(246, 240, 255);")
        self.frame_53.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_53.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_53.setObjectName("frame_53")
        self.trText_6 = QtWidgets.QPlainTextEdit(self.frame_53)
        self.trText_6.setEnabled(False)
        self.trText_6.setGeometry(QtCore.QRect(40, 50, 691, 181))
        self.trText_6.setObjectName("trText_6")
        self.label_47 = QtWidgets.QLabel(self.frame_53)
        self.label_47.setGeometry(QtCore.QRect(40, 20, 101, 21))
        self.label_47.setStyleSheet("font: 11pt \"Arial\";")
        self.label_47.setObjectName("label_47")
        self.verticalLayout_6.addWidget(self.frame_53, 0, QtCore.Qt.AlignBottom)
        self.tabWidget.addTab(self.tab_7, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 775, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(4)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_3.setText(_translate("MainWindow", "İngilizce"))
        self.label_2.setText(_translate("MainWindow", "Türkçe"))
        self.label_6.setText(_translate("MainWindow", "İngilizce"))
        self.translateBtn.setText(_translate("MainWindow", "Çevir"))
        self.label_7.setText(_translate("MainWindow", "Türkçe"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Kendi Modelimiz"))
        self.label_9.setText(_translate("MainWindow", "İngilizce"))
        self.label_11.setText(_translate("MainWindow", "Türkçe"))
        self.label_13.setText(_translate("MainWindow", "İngilizce"))
        self.translateBtn_2.setText(_translate("MainWindow", "Çevir"))
        self.label_14.setText(_translate("MainWindow", "Helsinki-NLP/opus-mt-tc-big-en-tr ile"))
        self.label_15.setText(_translate("MainWindow", "Türkçe"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Hugging Face 1"))
        self.label_25.setText(_translate("MainWindow", "İngilizce"))
        self.label_27.setText(_translate("MainWindow", "Türkçe"))
        self.label_29.setText(_translate("MainWindow", "İngilizce"))
        self.translateBtn_4.setText(_translate("MainWindow", "Çevir"))
        self.label_30.setText(_translate("MainWindow", "Helsinki-NLP/opus-tatoeba-en-tr ile"))
        self.label_31.setText(_translate("MainWindow", "Türkçe"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("MainWindow", "Hugging face 2"))
        self.label_33.setText(_translate("MainWindow", "Türkçe"))
        self.label_35.setText(_translate("MainWindow", "İngilizce"))
        self.label_37.setText(_translate("MainWindow", "Türkçe"))
        self.translateBtn_5.setText(_translate("MainWindow", "Çevir"))
        self.label_38.setText(_translate("MainWindow", "Helsinki-NLP/opus-mt-tr-en ile"))
        self.label_39.setText(_translate("MainWindow", "İngilizce"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_6), _translate("MainWindow", "Hugging face 3"))
        self.label_41.setText(_translate("MainWindow", "Türkçe"))
        self.label_43.setText(_translate("MainWindow", "İspanyolca"))
        self.label_45.setText(_translate("MainWindow", "Türkçe"))
        self.translateBtn_6.setText(_translate("MainWindow", "Çevir"))
        self.label_46.setText(_translate("MainWindow", "Helsinki-NLP/opus-mt-tr-es ile"))
        self.label_47.setText(_translate("MainWindow", "İspanyolca"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_7), _translate("MainWindow", "Es-Tr"))