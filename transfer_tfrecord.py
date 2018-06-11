import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math 





def getDatafile(file_dir, train_size, val_size):
    """Get list of train, val, test image path and label Parameters: 
    ----------- file_dir : str, file directory 
    train_size : float, size of test set 
    val_size : float, size of validation set 
    Returns: -------- train_img : str, list of train image path train_labels : int, list of train label test_img : test_labels : val_img : val_labels : """

    # images path list
    images_path = []
    # os.walk 遍历文件夹下的所有文件，包括子文件夹下的文件
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            images_path.append(os.path.join(root, name))

    # labels，images path have label of image
    labels = []
    for image_path in images_path:
        label = int(image_path.split('/')[-2]) # 将对应的label提取出来
        labels.append(label)

    # 先将图片路径和标签合并
    temp = np.array([images_path, labels]).transpose()
    # 提前随机打乱
    np.random.shuffle(temp)

    images_path_list = temp[:, 0]    # image path
    labels_list = temp[:, 1]         # label

    # train val test split
    train_num = math.ceil(len(temp) * train_size)
    val_num = math.ceil(len(temp) * val_size)

    # train img and labels
    train_img = images_path_list[0:train_num]
    train_labels = labels_list[0:train_num]
    train_labels = [int(float(i)) for i in train_labels]

    # val img and labels
    val_img = images_path_list[train_num:train_num+val_num]
    val_labels = labels_list[train_num:train_num+val_num]
    val_labels = [int(float(i)) for i in val_labels]

    # test img and labels
    test_img = images_path_list[train_num+val_num:]
    test_labels = labels_list[train_num+val_num:]
    test_labels = [int(float(i)) for i in test_labels]

    # 返回图片路径列表和对应标签列表
    return train_img, train_labels, val_img, val_labels, test_img, test_labels

def convert_to_TFRecord(images, labels, save_dir, name):
    """Convert images and labels to TFRecord file. Parameters: ----------- images : list of image path, string labels : list of labels, int save_dir : str, the directory to save TFRecord file name : str, the name of TFRecord file Returns: -------- no return """

    filename = os.path.join(save_dir, 'cache', name + '.tfrecords')
    n_samples = len(labels)

    if np.shape(images)[0] != n_samples:
        raise ValueError('Images size {} does not match label size {}'.format(images.shape[0], n_samples))

    writer = tf.python_io.TFRecordWriter(filename)       # TFRecordWriter class
    print('Convert to TFRecords...' )
    for i in xrange(0, n_samples):
        try:
            # 首先利用matplotlib读取图片，类型是np.ndarray(uint8)
            image = plt.imread(images[i])                # type(image) must be array
            image_raw = image.tobytes()                  # transform array to bytes
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                            'label': _int64_feature(label),
                            'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:{}'.format(images[i]))
            print('Skip it!')
    writer.close()
    print('Done' )

# Main
if __name__ == '__main__':
    # figure dir
    project_dir = os.getcwd()
    figure_dir = os.path.join(project_dir, 'dataset')

    # get list of images path and list of labels
    train_img, train_labels, val_img, val_labels, test_img, test_labels = getDatafile(figure_dir,
                                                                                      train_size=0.67,
                                                                                      val_size=0.1)
    # convert TFRecord file
    TFRecord_list = ['train', 'val', 'test']
    img_labels_list = [[train_img, train_labels], [val_img, val_labels], [test_img, test_labels]]
    save_dir = os.getcwd()
    for index, TFRecord_name in enumerate(TFRecord_list):
        convert_to_TFRecord(img_labels_list[index][0], img_labels_list[index][1],
                            save_dir,
                            TFRecord_name)