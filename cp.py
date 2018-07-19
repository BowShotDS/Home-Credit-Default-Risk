

# coding: utf-8
import tensorflow as tf
import json
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import  make_classification


# 常用函数定义
def weight_variable(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # 偏差初始化为0.1
    return tf.Variable(initial)

def conv2d1(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def conv2d2(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')

def squash(vector):
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)

def autoNorm(data):
    mins = data.min(0)
    maxs = data.max(0)
    ranges = maxs - mins
    normData = np.zeros(np.shape(data))
    row = data.shape[0]
    normData = data - np.tile(mins,(row,1))
    normData = normData / (np.tile(ranges,(row,1))+epsilon)
    return normData


#  读入数据
Batch_size = 100
epsilon = 1e-9

h=np.load("h.npy")
h1=np.nan_to_num(h)
h2=autoNorm(h1)
h3=np.nan_to_num(h2)

data1 = h3[:,2:122]
target = h3[:,1]
m,n = data1.shape
data=np.zeros([m,169],dtype=np.float32)
data[:,0:120]=data1
x_train,x_test,y_train,y_test =train_test_split(data,target,test_size=0.2)
y_tmp = tf.one_hot(y_train, depth=2, axis=1, dtype=tf.float32)

data_queues = tf.train.slice_input_producer([x_train, y_tmp])
X, Y = tf.train.shuffle_batch(data_queues, num_threads=8,
                              batch_size=Batch_size,
                              capacity=Batch_size * 64,
                              min_after_dequeue=Batch_size * 32,
                              allow_smaller_final_batch=False)



x = tf.placeholder(tf.float32,[None,169])
y = tf.placeholder(tf.float32,[None,2])
x_image = tf.reshape(x, [-1, 13, 13, 1])
keep_prob = tf.placeholder(tf.float32)



# 根据关系程度定制稀疏矩阵
# #######################




# 构建卷积层1
print('------3.1------')
W_conv1 = weight_variable([5, 5, 1, 256])
b_conv1 = bias_variable([256])
h_conv1 = tf.nn.relu(conv2d1(x_image,W_conv1) + b_conv1)
h_conv1 = tf.reshape(h_conv1,[-1, 9, 9, 256])

# 构建Capsule层1
print('------3.2------')
W_caps1 = weight_variable([3, 3, 256, 256])
b_caps1 = bias_variable([256])
h_caps1 = tf.nn.relu(conv2d2(h_conv1,W_caps1) + b_caps1)
h_caps1 = tf.reshape(h_caps1,[-1, 512, 8,1])

# 构建Capsule层2
print('------3.3------')
input_caps2 = tf.reshape(h_caps1,[-1, 512, 1,  8, 1])
b_IJ = tf.constant(np.zeros([Batch_size, 512, 2, 3, 1], dtype=np.float32))

input = input_caps2
W = weight_variable([1, 512, 6, 8, 1])
biases = bias_variable([1, 1, 2, 3, 1])
input = tf.tile(input, [1, 1, 6, 1, 1])
u_hat = tf.reduce_sum(W * input, axis=3, keepdims=True)
u_hat = tf.reshape(u_hat, shape=[-1, 512, 2, 3, 1])
u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

for r_iter in range(3):
    if r_iter == 0:
        c_IJ0 = tf.nn.softmax(b_IJ, axis=2)
    if r_iter == 2:
        c_IJ1 = tf.nn.softmax(b_IJ, axis=2)
        s_J = tf.multiply(c_IJ1, u_hat)
        s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
        s_J = tf.reshape(s_J, shape=[-1, 1, 1, 6, 1])
        v_J = squash(s_J)
        v_J1 = tf.reshape(v_J, shape=[-1, 1, 2, 3, 1])
    elif r_iter < 2:
        c_IJ = tf.nn.softmax(b_IJ, axis=2)
        s_J = tf.multiply(c_IJ, u_hat_stopped)
        s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
        s_J = tf.reshape(s_J, shape=[-1, 1, 1, 6, 1])
        v_J = squash(s_J)
        v_J = tf.reshape(v_J, shape=[-1, 1, 2, 3, 1])

        v_J_tiled = tf.tile(v_J, [1, 512, 1, 1, 1])
        u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
        # u_produce_v = u_hat_stopped * v_J_tiled
        b_IJ += u_produce_v

capsules = v_J1
# capsules = tf.squeeze(capsules, axis=1)
capsules = tf.reshape(capsules,[Batch_size, 2, 3,1])

# 选取---->>>   [Batch_siz,1,16,1]
print('------3.4------')
length_v = tf.sqrt(tf.reduce_sum(tf.square(capsules),axis=2, keepdims=True) + epsilon)
softmax_v = tf.nn.softmax(length_v, axis=1)
argmax_idx = tf.to_int32(tf.argmax(softmax_v, axis=1))
argmax_idx = tf.reshape(argmax_idx, shape=(Batch_size, ))
masked_v = []
for i in range(Batch_size):
    v = capsules[i][argmax_idx[i], :]
    masked_v.append(tf.reshape(v, shape=(1, 1, 3, 1)))
masked_v=tf.concat(masked_v, axis=0)

# 构建全连接层
print('------3.5------')
vector_j = tf.reshape(masked_v, shape=(Batch_size, -1))
fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=256,activation_fn=tf.nn.relu)
fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=256,activation_fn=tf.nn.relu)
decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=120,activation_fn=tf.nn.sigmoid)


# loss Function && Optimizer
max_l = tf.square(tf.maximum(0., 0.9 - length_v))
max_r = tf.square(tf.maximum(0., length_v - 0.1))
max_l = tf.reshape(max_l, shape=(Batch_size, -1))
max_r = tf.reshape(max_r, shape=(Batch_size, -1))
T_c = Y
L_c = T_c * max_l + 0.5 * (1 - T_c) * max_r
margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

orgin = tf.reshape(X[:,0:120], shape=(Batch_size, -1))
squared = tf.square(decoded - orgin)
reconstruction_err = tf.reduce_mean(squared)

total_loss = margin_loss + 0.0032 * reconstruction_err

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(total_loss)

# correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
# self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

acc=tf.equal(y,length_v)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())#就是这一行
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess,coord)
k=0
try:
    while not coord.should_stop():
        # print('------1------')
        example, label = sess.run([X, Y])
        k+=1
        if k > 300:
            print('------2------')
            break
        else:
            # print('------3------')
            sess.run(train_step, feed_dict={x: example, y: label})
            # print('------4------')
        if k % 10 == 0:
            print('------5------')
            print(sess.run(length_v, feed_dict={x: example, y: label}))
            print('------5.0------')
            print(sess.run(y, feed_dict={x: example, y: label}))
            # print('------5.1------')
            # print(sess.run(c_IJ1, feed_dict={x: example, y: label}))
            # print('------5.2------')
            # print(sess.run(v_J, feed_dict={x: example, y: label}))
            # print('------5.3------')
            # print(sess.run(capsules, feed_dict={x: example, y: label}))
            print('------5.4------')
            print(sess.run(total_loss, feed_dict={x: example, y: label}))
            print('------6------')
except tf.errors.OutOfRangeError:
    print('Done training')
finally:
    print('------7------')
    coord.request_stop()
    print('------8------')
coord.join(threads)
# sess.close()

# ################################################
# ################################################



















































