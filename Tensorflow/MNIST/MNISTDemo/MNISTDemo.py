import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist.mnist_deep import weight_variable, bias_variable
#from PIL import Image, ImageFilter

# def imageprepare():
#
#     file_name='E:\a.png'#导入自己的图片地址
#
#     im = Image.open(file_name).convert('L')
#
#     im.save("E:\b.png")
#     plt.imshow(im)
#     plt.show()
#     tv = list(im.getdata())
#
#     tva = [ (255-x)*1.0/255.0 for x in tv]
#     return tva

# 实现回归模型
# 使用的是 y=wx+b，softmax函数

mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])  # 占位符

W = tf.Variable(tf.zeros([784, 10]))  # 权重W，设置为变量可以使用，也可以修改
b = tf.Variable(tf.zeros([10]))  # 偏置量bias

y = tf.nn.softmax(tf.matmul(x, W) + b)  # 实际输出值，softmax函数

# 训练模型,引入代价函数或损失函数，一种常见的代价函数是交叉熵
y_ = tf.placeholder("float", [None, 10])  # 标准输出值

'''
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # 求和，交叉熵函数

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  # 0.01代表学习率,用梯度下降算法实现

init = tf.global_variables_initializer()  # 初始化创建的变量
sess = tf.Session()
sess.run(init)
'''


# 初始化操作
# 权重初始化
def weight_cvariable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置项初始化
def bias_vcariable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积和池化
# 使用1步长，0边距的模板
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])  # 卷积在每个5x5的patch中算出32个特征，前两个是patch的大小，接着是输入的通道数目，最后是输出的通道数目
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])  # 第二三维是对于图片的宽高，最后一维是标图片的颜色通道，灰度图是1，彩色图是3.

#池化层 1
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 将x_image和权重向量进行卷积，加上偏置项，用ReLU激活函数，进行maxpooling
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#池化层 2
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层,图片尺寸减小到7x7，加入一个有1024个神经元的全连接层，处理整个图片
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 减少过拟合，加入dropput，屏蔽神经元的输出和自动处理神经元输出值的scale
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 训练和评估模型

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 500 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d,training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy % g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    # prediction = tf.argmax(y_conv, 1)
    # predint = prediction.eval(feed_dict={x: [result], keep_prob: 1.0}, session=sess)
    # print(h_conv2)
    #
    # print('recognize result:')
    # print(predint[0])
'''
# 开始训练模型
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 评估模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # 输出值与标准值进行比较
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 将预测结果的布尔值转换为浮点数，再取平均值

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
'''
