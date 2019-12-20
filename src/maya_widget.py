# First, and before importing any Enthought packages, set the ETS_TOOLKIT
# environment variable to qt4, to tell Traits that we will use Qt.

from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item
from mayavi import mlab
from PyQt5 import QtWidgets, QtCore
import numpy as np
import time
import os
from reshaper import Reshaper
import utils
os.environ['ETS_TOOLKIT'] = 'qt4'


# A QSlider with its own ID, used to determine which PC it corresponds to
# Customized signal. Agment original valueChanged(int) with sliderID, and
# the min, max values of the slider
class IndexedQSlider(QtWidgets.QSlider):
  valueChangeForwarded = QtCore.pyqtSignal(int, int, int, int)
  def __init__(self, sliderID, orientation, parent=None):
    QtWidgets.QSlider.__init__(self, orientation, parent)
    self.sliderID = sliderID
    self.valueChanged.connect(self.valueChangeForwarder)

  ''' Emit coustomized valuechanged sigmal '''
  def valueChangeForwarder(self, val):
    self.valueChangeForwarded.emit(
      self.sliderID, val, self.minimum(), self.maximum())

class myAction(QtWidgets.QAction):
  myact = QtCore.pyqtSignal(int)
  def __init__(self, _id, *args):
    QtWidgets.QAction.__init__(self, *args)
    self._id = _id
    self.triggered.connect(self.emitSelect)

  def emitSelect(self):
    self.myact.emit(self._id)

class Visualization(HasTraits):
  scene = Instance(MlabSceneModel, ())
  @on_trait_change('scene.activated')
  def update_plot(self, v, f):
    mlab.clf()
    if not isinstance(v, str):
      mlab.triangular_mesh(v[:, 0], v[:, 1], v[:, 2], f)
  # the layout of the dialog screated
  view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
    height=200, width=250, show_label=False), resizable=True)

# The QWidget for rendering 3D shape
class MayaviQWidget(QtWidgets.QWidget):
  def __init__(self, parent):
    QtWidgets.QWidget.__init__(self, parent)
    layout = QtWidgets.QVBoxLayout(self)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    self.visualization = Visualization()

    # The edit_traits call will generate the widget to embed.
    self.ui = self.visualization.edit_traits(parent=self, kind='subpanel').control
    layout.addWidget(self.ui)
    self.ui.setParent(self)

    # models for shape representing
    self.bodies = {"female": Reshaper(label="female"), "male":Reshaper(label="male")}
    self.body = self.bodies["female"]
    self.flag_ = 0

    self.vertices = self.body.mean_vertex
    self.normals = self.body.normals
    self.facets = self.body.facets
    self.input_data = np.zeros((utils.M_NUM, 1))
    self.update()

  def update(self):
    [self.vertices, self.normals, self.facets] = \
        self.body.mapping(self.input_data, self.flag_)
    self.vertices = self.vertices.astype('float32')
    self.visualization.update_plot(self.vertices, self.facets)

  def select_mode(self, label="female", flag=0):
    self.body = self.bodies[label]
    self.flag_ = flag
    self.update()

  def sliderForwardedValueChangeHandler(self, sliderID, val, minVal, maxVal):
    x = val / 10.0
    self.input_data[sliderID] = x
    start = time.time()
    self.update()
    print(' [**] update body in %f s' % (time.time() - start))

  def save(self):
    utils.save_obj("result.obj", self.vertices, self.facets+1)
    output = np.array(utils.calc_measure(self.body.cp, self.vertices, self.facets))
    for i in range(0, utils.M_NUM):
      print("%s: %f" % (utils.M_STR[i], output[i, 0]))

  def predict(self, data):
    mask = np.zeros((utils.M_NUM, 1), dtype=bool)
    for i in range(0, data.shape[0]):
      if data[i, 0] != 0:
        data[i, 0] -= self.body.mean_measure[i, 0]
        data[i, 0] /= self.body.std_measure[i, 0]
        mask[i, 0] = 1
    self.input_data = self.body.get_predict(mask, data)
    self.update()
    measure = self.body.mean_measure + self.input_data*self.body.std_measure
    return [self.input_data, measure]
