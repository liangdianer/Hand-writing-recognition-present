import config as cf
import tensorflow as tf
import numpy as np
from PIL import Image
import os

class DATA():

    def get_branch_data(self, st, en):
        LEN = 0
        X = [None] * ((en - st + 10) * 2)
        Y = [None] * ((en - st + 10) * 2)

        for i in range(st, en):
            for k in range(2):
                if k == 0:
                    FILE_DIR = cf.DATA_PATH + '噪声训练集/' + chr(ord('A') + k) + '/'
                else:
                    FILE_DIR = cf.DATA_PATH + '噪声训练集/' + chr(ord('A') + (i % 4) + 1) + '/'

                FILE_PATH = FILE_DIR + str(i) + '.jpg'

                img = Image.open(FILE_PATH)
                img_arr = np.array(img)

                if (img_arr.shape == (32, 32)):
                    img_arr = img_arr.reshape((32, 32, 1))
                    X[LEN] = img_arr
                    if k == 0:
                        Y[LEN] = 1
                    else:
                        Y[LEN] = 0

                    LEN = LEN + 1

        X = X[: LEN]
        Y = Y[: LEN]

        X = np.array(X)
        Y = np.array(Y)

        X = X.reshape((-1,32,32,1))
        Y = Y.reshape((-1,1))

        return X, Y

    
    def get_train_data(self, size=11000):

        BRANCH_NUM = 1#int(size/(cf.BRANCH_SIZE * 5))

        X = [None] * BRANCH_NUM
        Y = [None] * BRANCH_NUM

        for k in range(BRANCH_NUM):
            print(k)
            if k == BRANCH_NUM - 1:
                X[k], Y[k] = self.get_branch_data(k * cf.BRANCH_SIZE, int(size/5))
            else:
                X[k], Y[k] = self.get_branch_data(k * cf.BRANCH_SIZE, (k + 1) * cf.BRANCH_SIZE)

        return X, Y


    def get_test_data(self):
        LEN = 0
        X = [None] * 6000
        Y = [None] * 6000

        for k in range(5):
            FILE_DIR = cf.DATA_PATH + '噪声验证集/' + chr(ord('A') + k) + '/'

            FILE_LIS = os.listdir(FILE_DIR)

            for FILE_PATH_1 in FILE_LIS:

                FILE_PATH = FILE_DIR + FILE_PATH_1

                img = Image.open(FILE_PATH)

                img_arr = np.array(img)

                if (img_arr.shape == (32, 32)):
                    X[LEN] = img_arr
                    Y[LEN] = [0]

                    if k == 0:
                        Y[LEN] = 1
                    else:
                        Y[LEN] = 0

                    LEN = LEN + 1

        X = X[: LEN]
        Y = Y[: LEN]

        X = np.array(X)
        X = X.reshape((-1,32,32,1))
        Y = np.array(Y)
        Y = Y.reshape((-1,1))
        return X, Y
    

#测试DATA类
data = DATA()
X, Y = data.get_train_data()

x_test, y_test = data.get_test_data()
print(X[0].shape)
print(Y[0].shape)


class MODEL():
    def __init__(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 1], name='X_input')
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.data = DATA()
        self.L2 = None
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.x_test = None


    def assemble_graph(self, w_alpha = 0.01, b_alpha=0.01):
        #第一层
        #10个3*3*3的卷积核
        w_c1 = tf.Variable(w_alpha*tf.random_normal([5, 5, 1, 8]), dtype=tf.float32)
        b_c1 = tf.Variable(b_alpha*tf.random_normal([8]), dtype=tf.float32)

        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.X, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self.keep_prob)
        #shape为(size, 16, 16, 16)

        

        
        #第二层
        w_c2 = tf.Variable(w_alpha*tf.random_normal([5, 5, 8, 16]), dtype=tf.float32)
        b_c2 = tf.Variable(b_alpha*tf.random_normal([16]), dtype=tf.float32)

        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_prob)
        #shape为(size, 8, 8, 32)
        
        
        
        #第三层
        w_c3 = tf.Variable(w_alpha*tf.random_normal([2, 2, 16, 32]), dtype=tf.float32)
        b_c3 = tf.Variable(b_alpha*tf.random_normal([32]), dtype=tf.float32)

        conv3 = tf.nn.tanh(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_prob)
        #shape为(size, 4, 4, 64)

        
        #全连接层
        cnn_out = tf.reshape(conv3, [-1, 4 * 4 * 32])
        w_out = tf.Variable(w_alpha*tf.random_normal([4 * 4 * 32, 1]), dtype=tf.float32)
        b_out = tf.Variable(w_alpha*tf.random_normal([1]), dtype=tf.float32)

        Y_p =  tf.add(tf.matmul(cnn_out, w_out), b_out, name='before_sigmoid')
        #shape为(size, 5)


        w1 = tf.reduce_mean(tf.square(w_c1))
        b1 = tf.reduce_mean(tf.square(b_c1))

        w2 = tf.reduce_mean(tf.square(w_c2))
        b2 = tf.reduce_mean(tf.square(b_c2))

        w3 = tf.reduce_mean(tf.square(w_c3))
        b3 = tf.reduce_mean(tf.square(b_c3))

        wo = tf.reduce_mean(tf.square(w_out))
        bo = tf.reduce_mean(tf.square(b_out))

        self.L2 = w1 + b1 + w2 + b2 + w3 + b3 + wo + bo
        '''
        #测试用代码
        x,y = self.data.get_train_data()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run([conv2], feed_dict={self.X:x[0]})
            print(np.array(out).shape)
        '''
        return Y_p

    
    def get_acc(self, Y_p, Y):
        Y_p = Y_p[0]
        Y_p = np.array(Y_p)
        
        size, num = Y_p.shape
        
        ACC = 0
        SUM = size

        for k in range(size):
            if abs(Y_p[k] - Y[k]) < 0.5:
                ACC = ACC + 1
            else:
                '''
                img_arr = self.x_test[k]

                img_arr = img_arr.reshape(32, 32)
                img = Image.fromarray(img_arr).convert('L')
                img.show()
                print(' ')
                print(Y_p[k])
                print(Y[k])
                #print(img_arr.shape)
                '''
                pass

        return ACC/SUM 
        
        return 0
    
    def train(self, size = 11000, w_alpha = 0.01, b_alpha=0.01):
        #读入数据
        X, Y = self.data.get_train_data(size)
        BRANCH_NUM = len(X)
        

        x_test, y_test = self.data.get_test_data()

        self.x_test = x_test
        
        #获取神经网络输出与预测值
        
        network_output = self.assemble_graph(w_alpha, b_alpha)
        Y_p = tf.nn.sigmoid(network_output, name='Y_p')



        #计算loss
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=network_output, labels=self.Y)
        loss = tf.reduce_mean(loss)

        #设置op
        op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


        saver = tf.train.Saver()

        with tf.Session() as sess:
            #sess.run(tf.global_variables_initializer())
            saver.restore(sess, cf.MODEL_PATH + 'CNN.model-991')

            step = 0
            print('begin')
            
            while True: 
                '''
                _, L = sess.run([op,loss], feed_dict={self.X:X[step % BRANCH_NUM], self.Y:Y[step % BRANCH_NUM], self.keep_prob:0.5})
                
                if step % 20 == 0:
                    print(str(step) + '  ' + str(L))
                
                '''
                if step % 100 == 0:
                    y_p = sess.run([Y_p], feed_dict={self.X:x_test, self.keep_prob:1.0})
                    ypp = sess.run([Y_p], feed_dict={self.X:X[0], self.keep_prob:1.0})

                    acc_t = self.get_acc(ypp, Y[0])

                    acc = self.get_acc(y_p, y_test)
                    print('ACC:')
                    print(str(step) + ' ' + str(acc))
                    print(acc_t)
                
                
                if step % 1000 == 0:
                    saver.save(sess, cf.MODEL_PATH + 'CNN.model', global_step=step)
                
                step = step + 1
                return


model = MODEL()
model.train()
