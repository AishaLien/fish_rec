def getDatafile(file_dir, train_size, val_size):
    """Get list of train, val, test image path and label Parameters: ----------- file_dir : str, file directory train_size : float, size of test set val_size : float, size of validation set Returns: -------- train_img : str, list of train image path train_labels : int, list of train label test_img : test_labels : val_img : val_labels : """

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