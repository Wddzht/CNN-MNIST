import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(r"E:\_Python\MNIST\tmp\mnist", one_hot=True)


def train_size(num):
    print('Total ' + str(mnist.train.images.shape))





def display_digit(num):
    x_train = mnist.train.images[:55000, :]
    y_train = mnist.train.labels[:55000, :]
    print(y_train[num])
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28, 28])
    plt.title('Example: %d Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()


def display_mult_flat(start, stop):
    x_train = mnist.train.images[:55000, :]
    y_train = mnist.train.labels[:55000, :]
    images = x_train[start].reshape([1, 784])
    for i in range(start + 1, stop):
        images = np.concatenate((images, x_train[i].reshape([1, 784])))
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.show()


display_digit(np.random.randint(0, 55000))
display_mult_flat(0, 100)
