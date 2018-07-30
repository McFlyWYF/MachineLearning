'''
#没有数据则会从网站下载
mnist = input_data.read_data_sets("/path/to/MNIST_data/",one_hot = True)

#打印训练数据大小
print("Training data size: ",mnist.train.num_examples)

#打印测试数据大小
print("Valodating data size:",mnist.validation.num_examples)

#打印测试数据
print("Testing data size:",mnist.test.num_examples)

#测试数据答案
print("Example training data:",mnist.train.images[0])

#打印答案标签
print("Example training data abel",mnist.train.labels[0])

batch_size = 100
xs,ys = mnist.train.next_batch(batch_size)

print("X shape:",xs.shape)
print("Y shape:",ys.shape)
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784  # 输入层节点数
OUTPUT_NODE = 10  # 输出层的节点数

# 配置神经网络的参数
LAYER1_NODE = 500  # 隐藏层节点数
BATCH_SIZE = 100  # 一个训练batch中的训练数据大小

LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率

REGULARIZATION_RATE = 0.0001  # 正则化的系数
TRAINING_STEPS = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


# 辅助函数，计算前向传播结果
def inference(input_tensor,avg_class,weights1,weights2,biases1,biases2):

    # 没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        # 计算隐藏层的前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        # 计算输出层的前向传播结果，可以不加入softmax函数
        return tf.matmul(layer1, weights2) + biases2

    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 训练模型
def train():
    mnist = input_data.read_data_sets("/path/to/MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算前向传播算法
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练的轮数
    global_step = tf.Variable(0, trainable=False)

    # 初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量使用滑动平均
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用滑动平均之后的前向传播结果
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    # 交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 计算模型的正则化损失
    regularization = regularizer(weights1) + regularizer(weights2)

    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率
    leaining_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(leaining_rate).minimize(loss, global_step=global_step)

    # 更新参数和滑动平均值
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    # 将布尔值转换为实数型，再计算平均值,平均值就是模型在这组数据的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 验证数据
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        # 测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 迭代训练
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s),validation_accuracy using average model is %g" % (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 训练结束后检测正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step,test accuracy using average""model is %g" % (TRAINING_STEPS, test_acc))


# 主函数
if __name__ == "__main__":
    train()
