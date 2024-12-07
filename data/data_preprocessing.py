import os
import struct
import numpy as np
import matplotlib.pyplot as plt

# 读取标签数据集
def read_labels(file_path):
    with open(file_path, 'rb') as lbpath:
        labels_magic, labels_num = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    return labels

# 读取图片数据集
def read_images(file_path):
    with open(file_path, 'rb') as imgpath:
        images_magic, images_num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(images_num, rows * cols)
    return images, rows, cols


if __name__ == "__main__":
    # 读取训练集和测试集
    train_labels = read_labels('./Dataset/MNIST/train-labels-idx1-ubyte')
    train_images, train_rows, train_cols = read_images('./Dataset/MNIST/train-images-idx3-ubyte')
    test_labels = read_labels('./Dataset/MNIST/t10k-labels-idx1-ubyte')
    test_images, test_rows, test_cols = read_images('./Dataset/MNIST/t10k-images-idx3-ubyte')

    # 打印数据信息
    print(f'train_labels.shape: {train_labels.shape}')
    print(f'train_images.shape: {train_images.shape}')
    print(f'test_labels.shape: {test_labels.shape}')
    print(f'test_images.shape: {test_images.shape}')

    # 数据归一化到 [0, 1] 范围
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 测试取出一张图片和对应标签
    choose_num = 1  # 指定一个编号，你可以修改这里
    label = train_labels[choose_num]
    image = train_images[choose_num].reshape(train_rows, train_cols)

    # 显示该图片
    plt.imshow(image, cmap='gray')
    plt.title(f'The label is: {label}')
    plt.show()
