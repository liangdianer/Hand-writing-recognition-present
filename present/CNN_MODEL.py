import tensorflow as tf
import numpy as np
from PIL import Image
import os
import config as cf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(suppress=True)

class CNN_MODEL_ABCDE(object):
    def __init__(self):
        self.ACC = 0.9812

        self.graph=tf.Graph()

        with self.graph.as_default():
            saver = tf.train.import_meta_graph(cf.MODEL_PATH + 'CNN.model-ABCDE.meta')


        self.sess = tf.Session(graph=self.graph)


        with self.sess.as_default():
            with self.graph.as_default():
                saver.restore(self.sess, cf.MODEL_PATH + 'CNN.model-ABCDE')

        self.x_input = self.graph.get_tensor_by_name('X_input:0')
        self.Y_p = self.graph.get_tensor_by_name('Y_p:0')
        self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')


    def use_model(self, img):
        x = np.array(img)
        if (x.shape != (32, 32)):
            return -1

        x = x.reshape([32, 32, 1])

        X = [x]

        Y_p = self.sess.run([self.Y_p], feed_dict={self.x_input:X, self.keep_prob:1.0})
        
        Y_p = np.array(Y_p)[0][0]
        return Y_p


    #析构函数，用于归还sess
    def __del__( self ):  
        self.sess.close()
   
class CNN_MODEL_A(object):
    def __init__(self):
        self.ACC = 0.991

        self.graph=tf.Graph()

        with self.graph.as_default():
            saver = tf.train.import_meta_graph(cf.MODEL_PATH + 'CNN.model-A.meta')


        self.sess = tf.Session(graph=self.graph)


        with self.sess.as_default():
            with self.graph.as_default():
                saver.restore(self.sess, cf.MODEL_PATH + 'CNN.model-A')

        self.x_input = self.graph.get_tensor_by_name('X_input:0')
        self.Y_p = self.graph.get_tensor_by_name('Y_p:0')
        self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')


    def use_model(self, img):
        x = np.array(img)
        if (x.shape != (32, 32)):
            return -1

        x = x.reshape([32, 32, 1])

        X = [x]

        Y_p = self.sess.run([self.Y_p], feed_dict={self.x_input:X, self.keep_prob:1.0})
        
        Y_p = np.array(Y_p)[0][0]
        return Y_p


    #析构函数，用于归还sess
    def __del__( self ):  
        self.sess.close()
        return



class CNN_MODEL_C(object):
    def __init__(self):
        self.ACC = 0.992

        self.graph=tf.Graph()

        with self.graph.as_default():
            saver = tf.train.import_meta_graph(cf.MODEL_PATH + 'CNN.model-C.meta')


        self.sess = tf.Session(graph=self.graph)


        with self.sess.as_default():
            with self.graph.as_default():
                saver.restore(self.sess, cf.MODEL_PATH + 'CNN.model-C')

        self.x_input = self.graph.get_tensor_by_name('X_input:0')
        self.Y_p = self.graph.get_tensor_by_name('Y_p:0')
        self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')


    def use_model(self, img):
        x = np.array(img)
        if (x.shape != (32, 32)):
            return -1

        x = x.reshape([32, 32, 1])

        X = [x]

        Y_p = self.sess.run([self.Y_p], feed_dict={self.x_input:X, self.keep_prob:1.0})
        
        Y_p = np.array(Y_p)[0][0]
        return Y_p


    #析构函数，用于归还sess
    def __del__(self):  
        self.sess.close()
        return


class CNN_MODEL_E(object):
    def __init__(self):
        self.ACC = 0.989

        self.graph=tf.Graph()

        with self.graph.as_default():
            saver = tf.train.import_meta_graph(cf.MODEL_PATH + 'CNN.model-E.meta')


        self.sess = tf.Session(graph=self.graph)


        with self.sess.as_default():
            with self.graph.as_default():
                saver.restore(self.sess, cf.MODEL_PATH + 'CNN.model-E')

        self.x_input = self.graph.get_tensor_by_name('X_input:0')
        self.Y_p = self.graph.get_tensor_by_name('Y_p:0')
        self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')


    def use_model(self, img):
        x = np.array(img)
        if (x.shape != (32, 32)):
            return -1

        x = x.reshape([32, 32, 1])

        X = [x]

        Y_p = self.sess.run([self.Y_p], feed_dict={self.x_input:X, self.keep_prob:1.0})
        
        Y_p = np.array(Y_p)[0][0]
        return Y_p


    #析构函数，用于归还sess
    def __del__(self):  
        self.sess.close()
        return



class CNN_MODEL(object):
    def __init__(self):
        self.ABCDE = CNN_MODEL_ABCDE()

        self.MODEL = [None] * 5
        self.MODEL[0] = CNN_MODEL_A()
        self.MODEL[2] = CNN_MODEL_C()
        self.MODEL[4] = CNN_MODEL_E()
        #self.CDE = CNN_MODEL_CDE()
        

    def use_model(self, img):
        x = np.array(img)
        if (x.shape != (32, 32)):
            return -1

        x = x.reshape([32, 32, 1])

        X = [x]

        Y_p_ABCDE = self.ABCDE.sess.run([self.ABCDE.Y_p], feed_dict={self.ABCDE.x_input:X, self.ABCDE.keep_prob:1.0})

        Y_p_ABCDE = Y_p_ABCDE[0][0].tolist()
        
        poi = Y_p_ABCDE.index(max(Y_p_ABCDE))


        if poi == 0:
            Y_p_poi = self.MODEL[poi].sess.run([self.MODEL[poi].Y_p], feed_dict={self.MODEL[poi].x_input:X, self.MODEL[poi].keep_prob:1.0})

            if (Y_p_poi[0] < 0.4):
                Y_p_ABCDE[poi] = 0.0


        if poi == 2:
            Y_p_poi = self.MODEL[poi].sess.run([self.MODEL[poi].Y_p], feed_dict={self.MODEL[poi].x_input:X, self.MODEL[poi].keep_prob:1.0})

            if (Y_p_poi[0] < 0.2):
                Y_p_ABCDE[poi] = 0.0


        if poi == 4:
            Y_p_poi = self.MODEL[poi].sess.run([self.MODEL[poi].Y_p], feed_dict={self.MODEL[poi].x_input:X, self.MODEL[poi].keep_prob:1.0})

            if (Y_p_poi[0] < 0.2):
                Y_p_ABCDE[poi] = 0.0
        
        return Y_p_ABCDE


#注意，图片必须是32 × 32的灰度图片



if __name__ == '__main__':
    MODEL = CNN_MODEL_E()

    img = Image.open('/home/ffb/Workspace/Python-srf/手写字识别/噪声验证集/E/2730.jpg')

    print(MODEL.use_model(img))
