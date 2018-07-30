import tensorflow as tf
import input_data


BATCH_SIZE = 32


def weights(shape):
    i = tf.truncated_normal(stddev=0.1, shape=shape)#stddev 标准差，mean 平均值，shape 张量维度，这个函数产生正态分布
    return tf.Variable(i)

#偏移量
def biasses(shape):
    i = tf.constant(0.1, shape=shape)
    return tf.Variable(i)

#卷积层
def conv(image, filter):
    return tf.nn.conv2d(input=image, filter=filter, strides=[1, 1, 1, 1]
                        , padding="SAME")

#池化层
def max_pool_2x2(image):
    return tf.nn.max_pool(image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
                          , padding="SAME")

#网络层
def net(image, drop_pro):
    # [BATCH_SIZE , 28 , 28 , 1]
    w_conv1 = weights([5, 5, 1, 32])
    b_conv1 = biasses([32])
    conv1 = tf.nn.relu(conv(image, w_conv1) + b_conv1)  # [28,28]
    pool1 = max_pool_2x2(conv1)  # [14,14]

    w_conv2 = weights([5, 5, 32, 64])
    b_conv2 = biasses([64])
    conv2 = tf.nn.relu(conv(pool1, w_conv2) + b_conv2)  # [14,14]
    pool2 = max_pool_2x2(conv2)  # [7 , 7 ]

    image_daw = tf.reshape(pool2, shape=[-1, 7 * 7 * 64])  # [1,7*7*64]

    fc_w1 = weights(shape=[7 * 7 * 64, 1024])
    fc_b1 = biasses(shape=[1024])
    fc_1 = tf.nn.relu(tf.matmul(image_daw, fc_w1) + fc_b1)

    drop_out = tf.nn.dropout(fc_1, drop_pro)

    fc_2 = weights([1024, 10])
    fc_b2 = biasses([10])

    softmax = tf.nn.softmax(tf.matmul(drop_out, fc_2) + fc_b2)  # [BATCH_SIZE , 10]
    return softmax


def get_accuracy(logits, label):
    current = tf.cast(tf.equal(tf.arg_max(logits, 1), tf.arg_max(label, 1)), "float")
    accuracy = tf.reduce_mean(current)
    return accuracy


def train():
    mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)

    x = tf.placeholder("float", shape=[None, 784])  # 784 = 28*28
    y = tf.placeholder("float", shape=[None, 10])  # 正确的label
    drop_pro = tf.placeholder("float")

    images = tf.reshape(x, shape=[BATCH_SIZE, 28, 28, 1])

    logits = net(images, drop_pro)

    getAccuracy = get_accuracy(logits, y)
    cross_entropy = -tf.reduce_sum(y * tf.log(logits))

    global_step = tf.Variable(0, name="global_step")
    train_op = tf.train.GradientDescentOptimizer(0.001) \
        .minimize(cross_entropy, global_step=global_step)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(3000):
        batch = mnist.train.next_batch(BATCH_SIZE)
        if i % 100 == 0:
            accuracy = sess.run(getAccuracy, feed_dict={x: batch[0]
                , y: batch[1], drop_pro: 1})
            print("----step:%d-----accuracy:%g" % (i, accuracy))
        sess.run(train_op, feed_dict={x: batch[0]
            , y: batch[1], drop_pro: 0.5})

if __name__ == "__main__":
    train()