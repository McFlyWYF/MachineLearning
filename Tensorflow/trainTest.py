#训练神经网络的步骤

import tensorflow as tf

# 生成模拟数据集
from numpy.random import RandomState

# 第一步：定义神经网络结构和前向传播的输出结果
# 定义训练数据batch的大小
batch_size = 8

# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 在shape的一个维度上使用None，可以使用不大的batch大小

x = tf.placeholder(tf.float32, shape=(None, 2), name='x_input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')

# 定义神经网络前向传播,非线性模型
biases1 = tf.constant(-0.5)
biases2 = tf.constant(0.1)
a = tf.nn.relu(tf.matmul(x, w1) + biases1)
y = tf.nn.relu(tf.matmul(a, w2) + biases2)


#第二步：定义损失函数以及选择反向传播优化的方法
# 定义损失函数和反向传播算法
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))#交叉熵
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

# 定义规则给出样本的标签
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

#第三步：生成会话，并且在训练数据上反复运行反向传播的优化算法
# 创建一个会话来运行
sess = tf.Session()

init_op = tf.global_variables_initializer()
sess.run(init_op)
print('训练前的权值：')
print(sess.run(w1))
print(sess.run(w2))

# 设置训练的次数
step = 5000
for i in range(step):
    # 每次选取batch_size个样本进行训练
    start = (i * batch_size) % dataset_size
    end = min(start + batch_size, dataset_size)

    # 通过选取的样本训练神经网络并更新参数
    sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
    if i % 100 == 0:
        # 每隔一段时间计算在所有数据上的交叉熵并输出
        total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})

        print("After %d training step(s),cross entropy on all data is %g" % (i, total_cross_entropy))

print('训练之后的权值:')
print (sess.run(w1))
print (sess.run(w2))
sess.close()

