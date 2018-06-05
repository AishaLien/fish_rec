import tensorflow as tf
import numpy as np

def weight_variable(name,shape):
    #輸入張量(data) 並且 隨機依照標準差為0.1來設定初始重量
    #initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name,shape)

def bias_variable(name,shape):
    #誤差定義為常數(constant) 設定初始誤差
    #initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name,shape)
#### 變數定義
W_conv1 = weight_variable("W_conv1",[5,5, 1,32]) # filter 5x5, in size 1, out size 32
b_conv1 = bias_variable("b_conv1",[32])
W_conv2 = weight_variable("W_conv2 ",[5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable("b_conv2",[64])
W_fc1 = weight_variable("W_fc1",[7*7*64, 1024])
b_fc1 = bias_variable("b_fc1",[1024])
W_fc2 = weight_variable("W_fc2",[1024, 10])
b_fc2 = bias_variable("b_fc2",[10])

saver = tf.train.Saver()
with tf.Session() as sess:
    # 提取变量
    saver.restore(sess,"/set_model/model.ckpt")
    print("weights:", sess.run(W_conv1))
    #print("biases:", sess.run(W_conv2))
    ****graph.get_tensor_by_name