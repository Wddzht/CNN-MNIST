import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets(r'E:\_Python\MNIST\tmp\mnist', one_hot=True)
model_path = r'E:\_Python\MNIST\model\model'
# Paramaters
learning_rate = 0.001
training_iters = 100
batch_size = 16
display_step = 5

n_input = 784
n_classes = 10
dropout = 0.85

# 定义两个占位符来存储预测值和真实标签
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)


# 定义卷积层
def conv2d(x, W, b, strides=1):
    """
    卷积的函数
    :param x:指需要做卷积的输入图像，它要求是一个Tensor，
        具有[batch, in_height, in_width, in_channels]这样的shape，
        具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，
        注意这是一个4维的Tensor，要求类型为float32和float64其中之一

    :param W:相当于CNN中的卷积核，它要求是一个Tensor，
        具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
        具体含义是[卷积核的高度，卷积核的宽度，图像通道数，输出通道数(卷积核个数)]，
        要求类型与参数input相同，第三维in_channels，就是参数 input:x 的第四维

    :param b:
    :param strides: 卷积时在图像每一维的步长，这是一个一维的向量，长度4
        strides=[1, strides, strides, 1]
    :return:
        tf.nn.conv2d() 结果返回一个Tensor，即feature map，
        shape仍然是[batch, height, width, channels]。
    """
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    """
    最大池化

    tf.nn.max_pool(value, ksize, strides, padding)
    value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，
        shape=[batch, height, width, channels]
    ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，
        如果不在batch和channels上做池化，这两个维度设为1
    strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
    padding：和卷积类似，可以取'VALID' 或者'SAME'
        `SAME`表示在卷积操作时，以0填充图像周围，在池化时0填充不够的行或列.
    """
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(x, weights, biases, dropout):
    # reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    # Reshape the conv2 to match the input of fully connected layer
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])

    fc1 = tf.matmul(fc1, weights['wd1'])
    fc1 = tf.add(fc1, biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Drouout
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.matmul(fc1, weights['out'])
    out = tf.add(out, biases['out'])
    return out


# weight ~= 5*5 的卷积核,需要被训练. 它要求是一个Tensor,
# shape=[filter_height, filter_width, in_channels, out_channels]
#     具体含义是[卷积核的高度, 卷积核的宽度, 图像通道数, 输出通道数].
#     要求类型与参数input相同, 有一个地方需要注意.
#     第三维in_channels, 就是参数 input: x的第四维 or 上一卷积层的out_channels .
#
# 全连接层的输入维度：
#     28*28 --conv1('SAME')--> 28*28*32 --pool1('SAME')--> 14*14*32
#     14*14*32 --conv2('SAME')--> 14*14*64  --pool2('SAME')--> 7*7*64
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = conv_net(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

train_loss = []  # 记录每次迭代的损失
train_acc = []
test_acc = []
saver = tf.train.Saver(max_to_keep=5)
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step <= training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            loss_train, acc_train = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            print('Iter {:}, Minibatch Loss= {:.2f}, Training Accuracy= {:.2f}'.format(step, loss_train, acc_train))
            acc_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.})
            print('Test Accuracy= {:.2f}'.format(acc_test))

            train_loss.append(loss_train)
            train_acc.append(acc_train)
            test_acc.append(acc_test)
        if step % 20 == 0:
            saver.save(sess, model_path, step)
        step += 1
    saver.save(sess, model_path, step)

eval_indices = range(0, training_iters, display_step)
plt.plot(eval_indices, train_loss)
plt.title('Softmax Loss per iteration')
plt.xlabel('Iteration')
plt.ylabel('Softmax Loss')
plt.show()

plt.plot(eval_indices, train_acc, label='train accuracy')
plt.plot(eval_indices, test_acc, label='test accuracy')
plt.title('train and test accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
