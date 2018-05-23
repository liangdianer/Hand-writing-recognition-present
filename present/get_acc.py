from CNN_MODEL import CNN_MODEL
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
        FILE_DIR = cf.DATA_PATH + '噪声验证集/' + chr(ord('A') + k) + '/'
        FILE_LIS = os.listdir(FILE_DIR)

        save = [0] * 5
        
        for FILE_PATH_1 in FILE_LIS: 

            summ = summ + 1
            FILE_PATH = FILE_DIR + FILE_PATH_1

            img = Image.open(FILE_PATH)

            img_arr = np.array(img)

            if (img_arr.shape == (32, 32)):
                Y_p = MODEL.use_model(img)#.tolist()
                poi = Y_p.index(max(Y_p))

                if poi == k:
                    true = true + 1
                else:
                    save[poi] = save[poi] + 1
                    pass

        #print(save)

    return true / summ


print(get_test_data())
#get_test_data()