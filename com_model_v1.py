import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 繪圖工具
import matplotlib.pyplot as plt
#工具定義
import numpy as np
#自定義
import read_data

# Parameters
model_path = "./set_model/model.ckpt"
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
	#輸入張量(data) 並且 隨機依照標準差為0.1來設定初始重量
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
	#誤差定義為常數(constant) 設定初始誤差
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')




########## 定義placeholder(給圖片預留位置) 這邊給予的是圖片訊息 ##########
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
#答案為0~9一共十組
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

#將圖片reshape， -1表示會自動算幾組
#圖片格式: -1為不管目前的維度,28*28的像素點，1/3分別代表黑白/彩色
x_image = tf.reshape(xs, [-1, 28, 28, 1])



##########   開始組裝神經網路   ##########

## conv1 layer ##

#1:表示 input_size  32:表示output_size 所以這裡表示一張圖總共訓練出32個filter
W_conv1 = weight_variable([5,5, 1,32]) # filter 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)    # output size 14x14x32

## conv2 layer ##

#這裡表示 一張圖訓練出2個filter
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

 

## func1 layer ##
#最後出來是64個 
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]

 

#這裡將第2層max_pool 過後的神經元 全部攤平 FLATTEN
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

 

## func2 layer(output) ##

#倒數第二層為1024個神經元 最後一層為10個神經元 採用softmax當成最後一層的激活函數
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

########## 定義loss function 以及 優化函數 ##########
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))# loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)



with tf.Session() as sess:
    with tf.device("/cpu:0"):
        ########## 定義Sess 以及初始化 ##########
        #sess = tf.Session()
        # 之後假如有save 則不再需要初始化
        sess.run(tf.initialize_all_variables())

        ########## 存取模型 並讓之後來套用 ##########
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver() 
        save_path = saver.save(sess, model_path)
        print("Model save to file: %s" % model_path)
         

        #開始訓練，dropout 0.5代表隨機隱藏掉一半神經元的資訊

        #科學家們發現這樣可以有效的減少overfitting

        #有關dropout的相關資訊可以參考這篇

        #本地圖片測試
        image, label = read_data.read_and_decode("train_data.tfrecords")
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        batch_size = 100
        example = np.zeros((batch_size,128,128,3))
        l = np.zeros((batch_size,1))

        #設定代數
        try:
            for i in range(100):
                #設定下一筆的訓練資料跑多少(這邊設定是100筆)
                batch_xs, batch_ys = mnist.train.next_batch(100)
                #訓練樣本打哪來
                sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
                if i % 50 == 0:
                    print(compute_accuracy(mnist.test.images, mnist.test.labels))
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)

        # Show image that we want to predict
        #plt.imshow(mnist.test.images[0].reshape((28, 28)))
        #plt.show()
        #ans = tf.argmax(prediction, 1)
        #print("Answer:", sess.run(ans, feed_dict={xs: mnist.test.images[0:1],ys: mnist.test.labels[0:1],keep_prob: 0.5}))

