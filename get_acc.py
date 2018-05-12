from CNN_MODEL import CNN_MODEL, CNN_MODEL_ABCDE
import config as cf
import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL = CNN_MODEL()

def get_test_data():
    summ = 0
    true = 0
    for k in range(5):
        print(k)
        FILE_DIR = cf.DATA_PATH + '训练集/' + chr(ord('A') + k) + '/'
        FILE_LIS = os.listdir(FILE_DIR)

        for FILE_PATH_1 in FILE_LIS:
            summ = summ + 1

            FILE_PATH = FILE_DIR + FILE_PATH_1

            img = Image.open(FILE_PATH)

            img_arr = np.array(img)

            if (img_arr.shape == (32, 32)):
                Y_p = MODEL.use_model(img)
                poi = Y_p.index(max(Y_p))

                if poi == k:
                    true = true + 1
                else:
                    '''
                    img.show()
                    print(Y_p)
                    print(k)
                    '''
                    pass

    return true / summ


print(get_test_data())