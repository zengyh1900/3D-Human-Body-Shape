import cv2
import numpy as np
import os

if __name__ == "__main__":
    path = "../pic/dl-w/"
    pic_list = os.listdir(path)
    print(pic_list)

    final = 0
    for pic in pic_list:
        img = cv2.imread(path + pic)
        img = img[170:-170, 180:-180]
        if (isinstance(final, int)):
            final = img
        else:
            final = np.column_stack((final, img))
        cv2.imshow("ans", final)
        cv2.waitKey()
    cv2.imwrite(path + "final.png", final)
