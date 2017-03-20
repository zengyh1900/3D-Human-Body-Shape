# First, and before importing any Enthought packages, set the ETS_TOOLKIT
# environment variable to qt4, to tell Traits that we will use Qt.

from myutils import *
from model import *
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item
from mayavi import mlab
import numpy
import time
import os
from PyQt4 import QtGui, QtCore
from PyQt4.QtOpenGL import *
from OpenGL.GL import *
from ctypes import *
os.environ['ETS_TOOLKIT'] = 'qt4'


# A QSlider with its own ID, used to determine which PC it corresponds to
# Customized signal. Agment original valueChanged(int) with sliderID, and
# the min, max values of the slider
class IndexedQSlider(QtGui.QSlider):
    valueChangeForwarded = QtCore.pyqtSignal(int, int, int, int)

    def __init__(self, sliderID, orientation, parent=None):
        QtGui.QSlider.__init__(self, orientation, parent)
        self.sliderID = sliderID
        self.connect(self, QtCore.SIGNAL('valueChanged(int)'),
                     self.valueChangeForwarder)

    ''' Emit coustomized valuechanged sigmal '''

    def valueChangeForwarder(self, val):
        self.valueChangeForwarded.emit(
            self.sliderID, val, self.minimum(), self.maximum())


class myAction(QtGui.QAction):
    myact = QtCore.pyqtSignal(int)

    def __init__(self, _id, *args):
        QtGui.QAction.__init__(self, *args)
        self._id = _id
        self.connect(self, QtCore.SIGNAL("triggered()"), self.emitSelect)

    def emitSelect(self):
        self.myact.emit(self._id)


class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())

    @on_trait_change('scene.activated')
    def update_plot(self, v, f):
        mlab.clf()
        mlab.triangular_mesh(v[:, 0], v[:, 1], v[:, 2], f)
    # the layout of the dialog screated
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False), resizable=True)


# The QWidget for rendering 3D shape
class MayaviQWidget(QtGui.QWidget):

    def __init__(self, parent, file):
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.visualization = Visualization()

        # The edit_traits call will generate the widget to embed.
        self.ui = self.visualization.edit_traits(
            parent=self, kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)

        # models for shape representing
        self.data = [MetaData(file, 1), MetaData(file, 2)]
        self.miner = [Miner(self.data[0]), Miner(self.data[1])]
        self.reshaper = [Reshaper(self.data[0]), Reshaper(self.data[1])]

        self.models = [MeasureModel(self.reshaper[0], self.reshaper[1]),
                       VertexModel(self.reshaper[0], self.reshaper[1]),
                       DeformModel(self.reshaper[0], self.reshaper[1]),
                       VertexGlobal(self.reshaper[0], self.reshaper[1]),
                       DeformGlobal(self.reshaper[0], self.reshaper[1]),
                       DeformLocal(self.reshaper[0], self.reshaper[1])]
        self.mode = ["Measure Model", "Vertex Model", "Deform Model",
                     "Vertex Global", "Deform Global", "Deform Local"]
        self.mode_index = 0
        self.flag_ = 1
        self.current_model = self.models[self.mode_index]
        self.current_model.set_body(self.flag_)
        body = self.current_model.current_body

        self.paras = self.current_model.paras
        self.ans_path = self.paras["ans_path"]
        self.input_data = numpy.zeros((body.m_num, 1))

        # data for Model
        self.deformation = None
        self.vertices = body.mean_vertex
        self.normals = body.normals
        self.facets = body.facet
        self.update()

    def update(self):
        [self.vertices, self.normals, self.facets] = \
            self.current_model.mapping(self.input_data)
        self.vertices = self.vertices.astype('float32')
        self.deformation = self.current_model.deformation
        self.visualization.update_plot(self.vertices, self.facets)

    def set_flag(self, flag):
        self.flag_ = flag
        self.current_model.set_body(flag)
        self.update()

    def select_mode(self, i):
        self.mode_index = i
        self.current_model = self.models[i]
        self.current_model.set_body(self.flag_)
        self.update()

    def sliderForwardedValueChangeHandler(self, sliderID, val, minVal, maxVal):
        x = val / 10.0
        self.input_data[sliderID] = x
        start = time.time()
        self.update()
        print(' [**] update body in %f s' % (time.time() - start))

    def save(self):
        filename = self.ans_path + "test.obj"
        body = self.current_model.current_body
        body.save_obj(filename, self.vertices, self.facets)
        m_ans = body.calc_measure(self.vertices)
        m_ans[0, 0] = (m_ans[0, 0]**3) / (1000**3)
        print(' [**] m_ans: ')
        for i in range(0, body.m_num):
            print("%s: %f" % (body.m_str[i], m_ans[i, 0]))

    def predict(self, data):
        body = self.current_model.current_body
        mask = numpy.zeros((body.m_num, 1), dtype=bool)
        for i in range(0, data.shape[0]):
            if data[i, 0] != 0:
                data[i, 0] -= body.mean_measure[i, 0]
                data[i, 0] /= body.std_measure[i, 0]
                mask[i, 0] = 1
        self.input_data = self.miner[self.flag_ - 1].get_predict(mask, data)
        self.update()
        return [self.input_data,
                body.mean_measure + self.input_data * body.std_measure]
