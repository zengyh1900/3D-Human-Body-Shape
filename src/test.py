from dataProcess.dataModel import *
from dataProcess.Masker import *
from Model.vertexGlobal import *
from Model.deformGlobal import *
from Model.deformLocal import *


if __name__ == "__main__":
    filename = "parameter.json"
    data = rawData(filename)
    bd = basisData(data)
    mark = Masker(data)
    model = dataModel(bd, mark)


    vg = vertexGlobal(model)
    vg.v_rebuild()

    dg = deformGlobal(model)
    dg.global_rebuild()

    dl = deformLocal(model)
    dl.local_rebuild()

    