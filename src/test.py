# from dataProcess.dataModel import *
# from dataProcess.Masker import *
# from Model.vertexGlobal import *
# from Model.deformGlobal import *
# from Model.deformLocal import *


# if __name__ == "__main__":
#     filename = "parameter.json"
#     data = rawData(filename)
#     bd = basisData(data)
#     mark = Masker(data)
#     model = dataModel(bd, mark)


#     vg = vertexGlobal(model)
#     vg.v_rebuild()

#     dg = deformGlobal(model)
#     dg.global_rebuild()

#     dl = deformLocal(model)
#     dl.local_rebuild()


import numpy as np

if __name__ == "__main__":
    num = np.array([i for i in range(0, 12)]).reshape(3, 4)
    print(num)
    mean_num = np.array(num.mean(axis=1)).reshape(num.shape[0], 1)
    print(mean_num)
    std_num = np.array(num.std(axis=1)).reshape(num.shape[0], 1)
    print(std_num)
    t_num = num - mean_num
    print(t_num)
    t_num /= std_num
    print(t_num)
