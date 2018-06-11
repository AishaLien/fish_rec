""" Convolutional Neural Network.
Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.05
num_steps = 1000
batch_size = 100
display_step = 10
channal = 3

# Network Parameters
num_input = 784*channal # MNIST data input (img shape: 28*28)
num_classes = 4 # MNIST total classes (0-9 digits)
dropout = 0.5 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

#picture_set
#filenames = ["/tfrecord/train_data.tfrecord", "/tfrecord/test_data.tfrecord"]
#train_dataset = tf.data.TFRecordDataset(filenames)
filename = './record/train_data.tfrecords'
filename_test = './record/test_data.tfrecords'

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
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, channal])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, channal, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.initialize_all_variables() #tf.global_variables_initializer() 



######圖片設定#######
image_batch , label_batch = read_and_decode(filename, batch_size)
test_image_batch,test_label_batch = read_and_decode(filename_test, batch_size)
# 轉換陣列的形狀
image_batch_train  = tf.reshape(image_batch, [-1, 28*28*channal])
# 把 Label 轉換成獨熱編碼
label_batch_train = tf.one_hot(label_batch, num_classes)

# 轉換陣列的形狀
image_batch_test = tf.reshape(test_image_batch, [-1, 28*28*channal])

# 把 Label 轉換成獨熱編碼
label_batch_test = tf.one_hot(test_label_batch,num_classes)



# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)


    # 建立執行緒協調器
    coord = tf.train.Coordinator()
    # 啟動文件隊列，開始讀取文件
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        for step in range(1, num_steps+1):
            batch_x, batch_y = sess.run([image_batch_train, label_batch_train])
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
            if step % display_step == 0 or step == 1 :
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y,
                                                                     keep_prob: 1})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

        print("Optimization Finished!")
        test_batch_x, test_batch_y = sess.run([image_batch_test, label_batch_test])
        # Calculate accuracy for 256 MNIST test images
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: test_batch_x,
                                          Y: test_batch_y,
                                          keep_prob: 1.0}))
    except tf.errors.OutOfRangeError:
         print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    # 結束後記得把文件名隊列關掉
    coord.join(threads)
    sess.close()