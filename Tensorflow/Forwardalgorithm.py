import tensorflow as tf

'''
x = tf.constant([[0.7,0.9]])
w1 = tf.constant([[0.2,0.1,0.4],[0.3,-0.5,0.2]])
w2 = tf.constant([[0.6],[0.1],[-0.2]])

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

with tf.Session() as sess:
    n = sess.run(y)
    print(n)

weight = tf.Variable(tf.random_normal([2,3],stddev=2))#stddev是指标准差为2

#随机数生成函数

tf.random_normal  正态分布
tf.truncated_normal  正态分布,偏离平均值超过2个，重新随机分配
tf.random_uniform  平均分布
tf.random_gamma  Gamma分布


#常数生成函数

tf.zeros  全0数组
tf.ones   全1数组
tf.fill([2,3],9)   全部为给定数字9的数组
tf.constant([1,2,3])   给定值1,2,3的常量


biases = tf.Variable(tf.zeros([3]))
'''

# 前向传播算法
# 声明两个变量
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 暂时将输入特征向量定义为一个常量，x是一个1x2的矩阵
#x = tf.placeholder(tf.float32, shape=(1, 2), name="input")

#优化代码，将输入的变量改为多个，将1x2的矩阵改为3x2的矩阵，就可以得到3个样例的结果
x = tf.placeholder(tf.float32,shape=(3,2),name="input")

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
# 初始化的改进方法
init_op = tf.global_variables_initializer()
sess.run(init_op)

# print(sess.run(y))
#print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))#通过设置占位符，可以随意指定输入值，通过feed_dict指定x的取值
#等同于 x = tf.constant([[0.7],[0.9]])
# x 是 3x2的矩阵
print(sess.run(y,feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))

# sess.run(w1.initializer)#初始化w1
# sess.run(w2.initializer)#初始化w2
#
# print(tf.all_variables())
# sess.close()
#
# tf.assign(w1, w2, validate_shape=False)  # 维度再运行时是可改变的

'''
#定义的损失函数，cross_entropy定义了真实值和预测值之间的交叉熵
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))

#定义学习率
learning_rate = 0.001

#定义反向传播算法来优化神经网络参数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

比较常用的优化方法
tf.train.GradientDescentOptimizer
tf.train.AdamOptimizer
tf.train.MomentumOptimizer
'''

