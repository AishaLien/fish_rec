import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 繪圖工具
import matplotlib.pyplot as plt

# Parameters
model_path = "./set_model/model.ckpt"
filename = './record/train_data.tfrecords'
filename_test = './record/test_data.tfrecords'

#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
num_classes = 2 # total classes
batch_size = 100
channal = 3
def read_and_decode(filename, batch_size): 
    # 建立文件名隊列
    filename_queue = tf.train.string_input_producer([filename], 
                                                    num_epochs=None)
    
    # 數據讀取器
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    # 數據解析
    img_features = tf.parse_single_example(
            serialized_example,
            features={ 'label'    : tf.FixedLenFeature([], tf.int64),
                       'img_raw': tf.FixedLenFeature([], tf.string), })
    
    image = tf.decode_raw(img_features['img_raw'], tf.uint8)
    image = tf.reshape(image, [28, 28,channal])
    
    label = tf.cast(img_features['label'], tf.int64)

    # 依序批次輸出 / 隨機批次輸出
    # tf.train.batch / tf.train.shuffle_batch
    image_batch, label_batch =tf.train.shuffle_batch(
                                 [image, label],
                                 batch_size=batch_size,
                                 capacity=10000 + 3 * batch_size,
                                 min_after_dequeue=1000)

    return image_batch, label_batch

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
xs = tf.placeholder(tf.float32, [None, 784*channal]) # 28x28
#答案為0~9一共十組
ys = tf.placeholder(tf.float32, [None,num_classes])
keep_prob = tf.placeholder(tf.float32)

#將圖片reshape， -1表示會自動算幾組
#圖片格式: -1為不管目前的維度,28*28的像素點，1/3分別代表黑白/彩色
x_image = tf.reshape(xs, [-1, 28, 28, channal])



##########   開始組裝神經網路   ##########

## conv1 layer ##

#1:表示 input_size  32:表示output_size 所以這裡表示一張圖總共訓練出32個filter
W_conv1 = weight_variable([5,5, channal,32]) # filter 5x5, in size 1, out size 32
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
W_fc2 = weight_variable([1024, num_classes])
b_fc2 = bias_variable([num_classes])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

########## 定義loss function 以及 優化函數 ##########
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))# loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

######圖片設定#######

####### train
image_batch , label_batch = read_and_decode(filename, batch_size)
# 轉換陣列的形狀
image_batch_train  = tf.reshape(image_batch, [-1, 28*28*channal])
# 把 Label 轉換成獨熱編碼
label_batch_train = tf.one_hot(label_batch, num_classes)

####### test
test_image_batch,test_label_batch = read_and_decode(filename_test, batch_size)
# 轉換陣列的形狀
image_batch_test = tf.reshape(test_image_batch, [-1, 28*28*channal])
# 把 Label 轉換成獨熱編碼
label_batch_test = tf.one_hot(test_label_batch,num_classes)


with tf.Session() as sess:
    with tf.device("/cpu:0"):
        ########## 定義Sess 以及初始化 ##########
        #sess = tf.Session()
        # 之後假如有save 則不再需要初始化
        sess.run(tf.initialize_all_variables())
        # 建立執行緒協調器
        coord = tf.train.Coordinator()
        # 啟動文件隊列，開始讀取文件
        threads = tf.train.start_queue_runners(coord=coord)
        ########## 存取模型 並讓之後來套用 ##########
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver() 
        save_path = saver.save(sess, model_path)
        print("Model save to file: %s" % model_path)
         

        #開始訓練，dropout 0.5代表隨機隱藏掉一半神經元的資訊

        #科學家們發現這樣可以有效的減少overfitting

        #有關dropout的相關資訊可以參考這篇
        try:
            #設定代數
            for i in range(10000):
                #設定下一筆的訓練資料跑多少(這邊設定是100筆)
                batch_x, batch_y = sess.run([image_batch_train, label_batch_train])
                test_batch_x, test_batch_y = sess.run([image_batch_test, label_batch_test])
                #訓練樣本打哪來
                sess.run(train_step, feed_dict={xs: batch_x, ys: batch_y, keep_prob: 0.5})
                if i % 100 == 0:
                    print(compute_accuracy(test_batch_x, test_batch_y ))
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        # 結束後記得把文件名隊列關掉
        coord.join(threads)
        sess.close()

