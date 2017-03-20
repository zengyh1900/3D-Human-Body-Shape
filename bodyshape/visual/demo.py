#!/usr/bin/python
# coding=utf-8

from .maya_widget import *
from PyQt4 import QtGui, QtCore


# usage: GUI for showing all models
class HumanShapeAnalysisDemo(QtGui.QMainWindow):

    def __init__(self, file):
        QtGui.QMainWindow.__init__(self)
        self.statusBar().showMessage("Hello there")
        self.mainBox = QtGui.QHBoxLayout()

        container = QtGui.QWidget()
        container.setWindowTitle("Embedding Mayavi in a PyQt4 Application")
        layout = QtGui.QGridLayout(container)
        self.viewer3D = MayaviQWidget(container, file)
        self.model = self.viewer3D.current_model
        self.body = self.model.current_body
        layout.addWidget(self.viewer3D, 1, 1)
        container.show()

        self.mainBox.addWidget(container)
        self.setWindowTitle(self.viewer3D.mode[self.viewer3D.mode_index])

        parentWidget = QtGui.QWidget()
        self.box = QtGui.QVBoxLayout()
        self.set_menu()
        self.set_radio()
        self.set_button()
        self.set_slider()

        self.mainBox.addLayout(self.box)
        self.mainBox.addWidget(self.viewer3D)
        self.set_dialog()
        self.resize(650, 650)

        self.viewer3D.setSizePolicy(
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        parentWidget.setLayout(self.mainBox)
        self.setCentralWidget(parentWidget)

    def set_menu(self):
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')

        exit = QtGui.QAction("Exit", self)
        exit.setShortcut("Ctrl+Q")
        exit.setStatusTip('Exit application')
        self.connect(exit, QtCore.SIGNAL(
            'triggered()'), QtCore.SLOT('close()'))
        fileMenu.addAction(exit)

        save = QtGui.QAction("Save", self)
        save.setShortcut("Ctrl+S")
        save.setStatusTip('save obj file')
        self.connect(save, QtCore.SIGNAL('triggered()'), self.viewer3D.save)
        fileMenu.addAction(save)

        for i in range(0, len(self.viewer3D.mode)):
            mode = myAction(i, self.viewer3D.mode[i], self)
            self.connect(mode, QtCore.SIGNAL('myact(int)'), self.select_mode)
            fileMenu.addAction(mode)
        self.setToolTip('This is a window, or <b>something</b>')

    def set_radio(self):
        self.radio1 = QtGui.QRadioButton('male')
        self.radio2 = QtGui.QRadioButton('female')
        self.radio1.setChecked(True)
        self.radio1.toggled.connect(self.radio_act)
        self.radio2.toggled.connect(self.radio_act)

        radio_box = QtGui.QHBoxLayout()
        radio_box.addWidget(self.radio1)
        radio_box.addWidget(self.radio2)
        self.box.addLayout(radio_box)

    def radio_act(self):
        if self.radio1.isChecked():
            self.viewer3D.set_flag(1)
        else:
            self.viewer3D.set_flag(2)

    def set_button(self):
        self.button_box = QtGui.QHBoxLayout()

        self.reset_button = QtGui.QPushButton("RESET")
        self.reset_button.setStatusTip('reset input to mean value')
        self.connect(self.reset_button, QtCore.SIGNAL(
            'clicked()'), self.reset)
        self.button_box.addWidget(self.reset_button)

        self.pre_button = QtGui.QPushButton("PREDICT")
        self.pre_button.setToolTip('model your own shape')
        self.connect(self.pre_button, QtCore.SIGNAL(
            'clicked()'), self.show_dialog)
        self.button_box.addWidget(self.pre_button)
        self.box.addLayout(self.button_box)

    def set_slider(self):
        self.slider = []
        self.spin = []
        self.label = []
        for i in range(0, self.body.m_num):
            hbox = QtGui.QHBoxLayout()
            slider = IndexedQSlider(i, QtCore.Qt.Horizontal, self)
            slider.setStatusTip('%d. %s' % (i, self.body.m_str[i]))
            slider.setRange(-30, 30)
            slider.valueChangeForwarded.connect(
                self.viewer3D.sliderForwardedValueChangeHandler)
            self.slider.append(slider)
            spinBox = QtGui.QSpinBox()
            spinBox.setRange(-30, 30)
            spinBox.valueChanged.connect(slider.setValue)
            slider.valueChanged.connect(spinBox.setValue)
            self.spin.append(spinBox)
            label = QtGui.QLabel()
            label.setText("PCA_%d" % (i + 1))
            label.setFixedWidth(150)
            self.label.append(label)
            hbox.addWidget(label)
            hbox.addWidget(slider)
            hbox.addWidget(spinBox)
            self.box.addLayout(hbox)
        for i in range(self.body.v_basis_num, self.body.m_num):
            self.label[i].setText(self.body.m_str[i])
            self.label[i].hide()
            self.slider[i].hide()
            self.spin[i].hide()

    def set_dialog(self):
        self.pre_dialog = QtGui.QDialog()
        self.dialogBox = QtGui.QVBoxLayout()
        self.pre_dialog.setWindowTitle("Synthesize your own body")
        self.editList = []
        for i in range(0, self.body.m_num):
            edit = QtGui.QLineEdit()
            self.editList.append(edit)
            label = QtGui.QLabel()
            label.setText(self.body.m_str[i])
            label.setFixedHeight(20)
            label.setFixedWidth(150)
            box = QtGui.QHBoxLayout()
            box.addWidget(label)
            box.addWidget(edit)
            self.dialogBox.addLayout(box)
        dialogOK = QtGui.QPushButton("OK")
        clearButton = QtGui.QPushButton("CLEAR")
        self.connect(dialogOK, QtCore.SIGNAL('clicked()'), self.predict)
        self.connect(clearButton, QtCore.SIGNAL('clicked()'), self.clear)
        box = QtGui.QHBoxLayout()
        box.addWidget(dialogOK)
        box.addWidget(clearButton)
        self.dialogBox.addLayout(box)
        self.pre_dialog.setLayout(self.dialogBox)

    def closeEvent(self, event):
        self.pre_dialog.close()
        event.accept()

    def reset(self):
        for i in range(0, self.viewer3D.current_model.demo_num):
            self.slider[i].setValue(0)

    def clear(self):
        for i in range(0, len(self.editList)):
            self.editList[i].clear()

    def show_dialog(self):
        self.pre_dialog.show()

    def select_mode(self, id):
        old_num = self.model.demo_num
        new_num = self.viewer3D.models[id].demo_num
        if old_num != new_num:
            if new_num == self.body.v_basis_num:
                for i in range(0, new_num):
                    self.label[i].setText("PCA_%d" % (i + 1))
                for i in range(new_num, old_num):  # 10~19
                    self.label[i].hide()
                    self.slider[i].hide()
                    self.spin[i].hide()
            else:
                for i in range(0, old_num):
                    self.label[i].setText(self.body.m_str[i])
                for i in range(old_num, new_num):  # 10~19
                    self.label[i].show()
                    self.slider[i].show()
                    self.spin[i].show()
        self.viewer3D.select_mode(id)
        self.setWindowTitle(self.viewer3D.mode[id])
        self.model = self.viewer3D.current_model
        self.body = self.model.current_body

    def set_flag(self, flag):
        self.viewer3D.set_flag(flag)
        self.body = self.model.current_body

    def predict(self):
        try:
            w = float(self.editList[0].text())
            h = float(self.editList[1].text())
            if self.model.demo_num != self.body.m_num:
                self.select_mode(3)
            data = []
            data.append(w ** (1.0 / 3.0) * 1000)
            data.append(h * 10)
            for i in range(2, len(self.editList)):
                try:
                    tmp = float(self.editList[i].text())
                    data.append(tmp * 10)
                except ValueError:
                    data.append(0)
            data = numpy.array(data).reshape(self.body.m_num, 1)
            [t_data, value] = self.viewer3D.predict(data)
            for i in range(2, len(self.editList)):
                self.editList[i].setText("%f" % (value[i, 0] / 10))
            for i in range(0, len(self.slider)):
                self.slider[i].valueChangeForwarded.disconnect(
                    self.viewer3D.sliderForwardedValueChangeHandler)
                self.slider[i].setValue(t_data[i] / 3.0 * 100.0)
                self.slider[i].valueChangeForwarded.connect(
                    self.viewer3D.sliderForwardedValueChangeHandler)
        except ValueError:
            self.editList[0].setText("Please input.")
            self.editList[1].setText("Please input.")
