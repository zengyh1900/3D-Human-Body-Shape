#!/usr/bin/python
# coding=utf-8


# usage: GUI for showing all models
class HumanShapeAnalysisDemo(QtGui.QMainWindow):

    def __init__(self, file):
        QtGui.QMainWindow.__init__(self)
        self.statusBar().showMessage("Hello there")
        self.mainBox = QtGui.QHBoxLayout()
        #--------------------------------
        container = QtGui.QWidget()
        container.setWindowTitle("Embedding Mayavi in a PyQt4 Application")
        # define a "complex" layout to test the behaviour
        layout = QtGui.QGridLayout(container)
        self.viewer3D = MayaviQWidget(container, file)
        layout.addWidget(self.viewer3D, 1, 1)
        container.show()
        self.mainBox.addWidget(container)
        #----------------------------------------
        # self.viewer3D = Viewer3DWidget(self, file)
        self.setWindowTitle(self.viewer3D.mode[self.viewer3D.modeIndex])
        # ===== set Menu ======
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
            self.connect(mode, QtCore.SIGNAL('myselect(int)'), self.selectMode)
            fileMenu.addAction(mode)
        self.setToolTip('This is a window, or <b>something</b>')

        # ===== create buttons and sliders ======
        createButtons = True
        if createButtons:
            parentWidget = QtGui.QWidget()
            # ---------------------------------------------------
            self.button_box = QtGui.QHBoxLayout()
            self.reset_button = QtGui.QPushButton("RESET")
            self.reset_button.setStatusTip('reset input to mean value')
            self.connect(self.reset_button, QtCore.SIGNAL(
                'clicked()'), self.reset)
            self.button_box.addWidget(self.reset_button)

            self.pre_button = QtGui.QPushButton("PREDICT")
            self.pre_button.setToolTip('model your own shape')
            self.connect(self.pre_button, QtCore.SIGNAL(
                'clicked()'), self.showDialog)
            self.button_box.addWidget(self.pre_button)
            # --------------------slider box--------------------------------
            self.slider = []
            self.spin = []
            self.label = []
            self.box = QtGui.QVBoxLayout()
            self.box.addLayout(self.button_box)
            for i in range(0, self.viewer3D.data.measure_num):
                hbox = QtGui.QHBoxLayout()
                slider = IndexedQSlider(i, QtCore.Qt.Horizontal, self)
                slider.setStatusTip('%d. %s' %
                                    (i, self.viewer3D.data.measure_str[i]))
                slider.setRange(-100, 100)
                slider.valueChangeForwarded.connect(
                    self.viewer3D.sliderForwardedValueChangeHandler)
                self.slider.append(slider)
                spinBox = QtGui.QSpinBox()
                spinBox.setRange(-100, 100)
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
            for i in range(self.viewer3D.basis.v_basis_num, self.viewer3D.data.measure_num):
                self.label[i].setText(self.viewer3D.data.measure_str[i])
                self.label[i].hide()
                self.slider[i].hide()
                self.spin[i].hide()
            self.mainBox.addLayout(self.box)
            self.mainBox.addWidget(self.viewer3D)
            # ============== create predict info dialog ==========
            self.pre_dialog = QtGui.QDialog()
            self.pre_dialog.setWindowTitle("Synthesize your own body")
            self.editList = []
            self.dialogBox = QtGui.QVBoxLayout()
            for i in range(0, self.viewer3D.data.measure_num):
                edit = QtGui.QLineEdit()
                self.editList.append(edit)
                label = QtGui.QLabel()
                label.setText(self.viewer3D.data.measure_str[i])
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
            #========================================================
            self.viewer3D.setSizePolicy(
                QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
            parentWidget.setLayout(self.mainBox)
            self.setCentralWidget(parentWidget)
        else:
            self.setCentralWidget(self.viewer3D)
        self.resize(650, 650)

    def closeEvent(self, event):
        self.pre_dialog.close()
        event.accept()

    def selectMode(self, id):
        old_num = self.viewer3D.current_model.demo_num
        new_num = self.viewer3D.models[id].demo_num
        if old_num != new_num:
            if new_num == self.viewer3D.basis.v_basis_num:
                for i in range(0, new_num):
                    self.label[i].setText("PCA_%d" % (i + 1))
                for i in range(new_num, old_num):  # 10~19
                    self.label[i].hide()
                    self.slider[i].hide()
                    self.spin[i].hide()
            else:
                for i in range(0, old_num):
                    self.label[i].setText(self.viewer3D.data.measure_str[i])
                for i in range(old_num, new_num):  # 10~19
                    self.label[i].show()
                    self.slider[i].show()
                    self.spin[i].show()
        self.viewer3D.selectMode(id)
        self.setWindowTitle(self.viewer3D.mode[id])

    def reset(self):
        for i in range(0, self.viewer3D.current_model.demo_num):
            self.slider[i].setValue(0)

    def clear(self):
        for i in range(0, len(self.editList)):
            self.editList[i].clear()

    def showDialog(self):
        self.pre_dialog.show()

    def predict(self):
        try:
            w = float(self.editList[0].text())
            h = float(self.editList[1].text())
            if self.viewer3D.current_model.demo_num != self.viewer3D.data.measure_num:
                self.selectMode(3)
            data = []
            data.append(w ** (1.0 / 3.0) * 1000)
            data.append(h * 10)
            for i in range(2, len(self.editList)):
                try:
                    tmp = float(self.editList[i].text())
                    data.append(tmp * 10)
                except ValueError:
                    data.append(0)
            [t_data, value] = self.viewer3D.predict(
                np.array(data).reshape(len(data), 1))
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


#####################################################################
#####################################################################
if __name__ == '__main__':
    file = "parameter.json"
    # mouse = [0, 0, -1, -1]
    # app = QtGui.QApplication(sys.argv)

    # win = HumanShapeAnalysisDemo(file)
    # win.show()
    # sys.exit(app.exec_())

    with open(filename, 'r') as f:
        paras = json.load(f)
    print(paras["part"])
