# First, and before importing any Enthought packages, set the ETS_TOOLKIT
# environment variable to qt4, to tell Traits that we will use Qt.
import os
import sys
sys.path.append("..")
from PyQt4 import QtGui, QtCore
from PyQt4.QtOpenGL import *
from OpenGL.GL import *
from ctypes import *

from Visual.trackball import *
from Model.measureModel import *
from Model.vertexModel import *
from Model.deformModel import *
from Model.vertexGlobal import *
from Model.deformGlobal import *
from Model.deformLocal import *

os.environ['ETS_TOOLKIT'] = 'qt4'
from pyface.qt import QtGui, QtCore
from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item
from mayavi import mlab
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor

################################################################################
#The actual visualization
class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())

    @on_trait_change('scene.activated')
    def update_plot(self, v, f):
        mlab.clf()
        mlab.triangular_mesh(v[:,0], v[:,1], v[:,2], f)
    # the layout of the dialog screated
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False),resizable=True)

################################################################################
# The QWidget containing the visualization, this is pure PyQt4 code.
class MayaviQWidget(QtGui.QWidget):
    ''' a class for rendering 3D models'''
    def __init__(self, parent, file):
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.visualization = Visualization()

        # If you want to debug, beware that you need to remove the Qt
        # input hook.
        #QtCore.pyqtRemoveInputHook()
        #import pdb ; pdb.set_trace()
        #QtCore.pyqtRestoreInputHook()

        # The edit_traits call will generate the widget to embed.
        self.ui = self.visualization.edit_traits(parent=self,kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)

        # data for Model
        self.deformation = None
        self.vertices = None
        self.normals = None
        self.faces = None

        # models for shape representing
        self.data = rawData(file)
        self.basis = basisData(self.data)
        self.miner = measureMining(self.data)
        self.mask = Masker(self.data)
        self.model = dataModel(self.basis, self.mask)
        self.models = [measureModel(self.model, self.miner), vertexModel(self.model), deformModel(self.model),
                       vertexGlobal(self.model), deformGlobal(self.model), deformLocal(self.model)]
        self.mode = ["Measure Model", "Vertex Model", "Deform Model",
                     "Vertex Global", "Deform Global", "Deform Local"]
        self.modeIndex = 0
        self.current_model = self.models[self.modeIndex]
        self.input_data = np.zeros((self.data.measure_num, 1))
        self.myupdata()

    def myupdata(self):
        [self.vertices, self.normals, self.faces] = self.current_model.mapping(self.input_data)
        self.vertices = self.vertices.astype('float32')
        self.deformation = self.current_model.deformation
        v = np.array(self.vertices.copy()).reshape(self.data.vertex_num,3)
        f = np.array(self.faces.copy()).reshape(self.data.face_num, 3)
        self.visualization.update_plot(v, f)

    def selectMode(self, i):
        self.modeIndex = i
        self.current_model = self.models[i]
        self.myupdata()

    def sliderForwardedValueChangeHandler(self, sliderID, val, minVal, maxVal):
        x = (val/100.0)*5.0
        self.input_data[sliderID] = x
        start = time.time()
        self.myupdata()
        print (' [**] update body in %f s'%(time.time()-start))

    def save(self):
        filename= self.model.ansPath + "test.obj"
        self.data.save_obj(filename, self.vertices, self.data.o_faces)
        output = np.array(self.data.calc_measures(self.vertices))
        print (' [**] output: ')
        for i in range(0, output.shape[0]):
            print ("%s: %f"%(self.data.measure_str[i], output[i,0]))

    def predict(self, data):
        mask = np.zeros((self.data.measure_num, 1), dtype=bool)
        for i in range(0, data.shape[0]):
            if data[i,0] != 0:
                data[i,0] = (data[i,0]-self.data.mean_measures[i,0])/self.data.std_measures[i,0]
                mask[i,0] = 1
        output = self.miner.getPredict(mask, data)
        self.input_data = np.array([c for c in output.flat]).reshape(self.data.measure_num, 1)
        self.myupdata()
        return [output, self.data.mean_measures+output*self.data.std_measures]
