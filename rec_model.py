#架構建立

#建立至少确保在2.1之前版本的Python可以正常运行一些新的语言特性
from __future__ import print_function
import os
# 只显示 warning 和 Error
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#引入tensorflow相關模組
import tensorflow as tf
# 繪圖工具
import matplotlib.pyplot as plt
#數據處理的工具
import numpy as np

#filter or kernal
def weight_variable(shape): 
    #輸入張量(data) 並且 隨機依照標準差為0.1來設定標準差 
    inital=tf.truncted_normal(shape,stddev=0.1)
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
    return tf.nn.conv2d(x,W,strides=[1,1,1,1]，padding='SAME') 
    return tf.Variable(initial)

#用pooling 減少訊息量(壓縮)
def max_poo_2x2(x):
    #x 接用conv2d輸出的東西，也就是圖片
    #將長寬圖形縮小，2*2 中取最大的
    # maxpooling or avgpooling 可以使用
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1])

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
h_pool = max_pool_2X2(h_conv1) #14*14*32 maxpool一次跨兩步

##cnn layer2 2次積捲
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)