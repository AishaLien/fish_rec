
#架構建立
#建立至少确保在2.1之前版本的Python可以正常运行一些新的语言特性
from __future__ import print_function
import os

# 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
#引入tensorflow相關模組
import tensorflow as tf
# 繪圖工具
import matplotlib.pyplot as plt
#數據處理的工具
import numpy as np
#debug
import h5py
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
#引入資料
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#filter or kernal
def weight_variable(shape): 
    #輸入張量(data) 並且 隨機依照標準差為0.1來設定標準差 
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#誤差修正
def bias_variable(shape): 
    #誤差定義為常數(constant)
    initial=tf.constant(0.1,shape=shape)
    #宣告initial為variable 
    return tf.Variable(initial)

def conv2d(x,W):
    #x為圖片的所有參數，weight此積捲的權重, 步長(隔多少取訊息)
    #stride[1, x_movement, y_movement,1] [0]&[3]都固定
    #padding 會抽取不同範圍 抽完之後valid 會變小 same 不會改變
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME') 
    return tf.Variable(initial)

#用pooling 減少訊息量(壓縮)
def max_pool_2x2(x):
    #x 接用conv2d輸出的東西，也就是圖片
    #將長寬圖形縮小，2*2 中取最大的
    # maxpooling or avgpooling 可以使用
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

#圖片訊息
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

#除新定義大小 -1 為不管目前的維度,28*28的像素點，1/3代表黑白/彩色
x_image = tf.reshape(xs,[-1,28,28,1])

## cnn layer1  1次積捲
#我們的kernal是5*5(長寬為5*5像素，高度為32(RGB疊起來的)) ，黑白圖片所以為1 , 輸出為32個特徵點
W_conv1 = weight_variable([5,5,1,32])
#bias的設置搭配輸出的特徵點
b_conv1 = bias_variable([32])
#第一次捲積(用conv2d處理 x_image*W_conv1+b_conv1，再進行非線性畫relu)
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)# output 為 28*28(使用same長寬保持)*32(深度變成32)
#進行pooling，h_pool是這層的輸出值
h_pool = max_pool_2x2(h_conv1) #14*14*32 maxpool一次跨兩步

##cnn layer2 2次積捲
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

##fully connected layer
#[n_samples,7,7,64]->>[n_samples,7*7*64]
#拉平積捲
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64]) 
#擴張輸入大小到輸出為1024
W_fc1=weight_variable([7*7*64,1024]) 
b_fc1=bias_variable([1024])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#排除overfitting
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
#要更改 輸出為可能的數量
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

#用softmax變成概率
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#衡量loss
cross_entropy=tf.reduce_mean(
    -tf.reduce_sum(ys*tf.log(prediction),
    reduction_indices=[1]))

train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.tabels))

#sess=tf.Session()
# tf.initialize_all_variables() 这种写法马上就要被废弃
# 替换成下面的写法:
#print(sess.run(tf.global_variables_initializer()))  
