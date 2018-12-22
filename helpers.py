import tensorflow as tf
from tensorflow import keras
from random import shuffle

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def print_length_and_shapes(train_images, train_labels, dev_images, dev_labels, test_images, test_labels):
    print('Debugging (Shapes)')
    print('__________________')
    print('train_images.shape')
    print(train_images.shape)
    print('len(train_labels)')
    print(len(train_labels))
    print('train_labels')
    print(train_labels)

    print('dev_images.shape')
    print(dev_images.shape)
    print('len(dev_labels)')
    print(len(dev_labels))
    print('dev_labels')
    print(dev_labels)

    print('test_images.shape')
    print(test_images.shape)
    print('len(test_labels)')
    print(len(test_labels))
    print('test_labels')
    print(test_labels)


def show_one_image(img):
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def show_multiple_images(images, labels, class_names):
    plt.figure(figsize=(10, 10))

    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])

    plt.show()


def reduce_fashion_mnist_dataset(images, labels):
    type_one = []
    type_two = []

    for i in labels:
        if labels[i] == 1:
            type_one.append(images[i])
        elif labels[i] == 2:
            type_two.append(images[i])

    labels_type_one = [0] * len(type_one)
    labels_type_two = [1] * len(type_two)

    type_one_data = list(zip(type_one, labels_type_one))
    type_two_data = list(zip(type_two, labels_type_two))

    data_mix = type_one_data + type_two_data
    shuffle(data_mix)

    data_images, data_labels = zip(*data_mix)
    data_images = np.array([np.array(img) for img in data_images])
    data_labels = np.array([np.array(label) for label in data_labels])

    return data_images, data_labels


def load_fashion_mnist_dataset():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images, train_labels = reduce_fashion_mnist_dataset(train_images, train_labels)
    test_images, test_labels = reduce_fashion_mnist_dataset(test_images, test_labels)

    return train_images, train_labels, test_images, test_labels
