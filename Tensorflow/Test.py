import tensorflow as tf

'''
# 使用张量计算中间结果
a = tf.constant([1.0, 2.0], name="a", dtype="float32")
b = tf.constant([2.0, 3.0], name="b", dtype="float32")
result1 = tf.add(a, b)
print(result1)
c = result1.get_shape()  # 使用get_shape()获取维度信息
print(c)

# 不使用张量计算中间结果
result2 = tf.constant([1.0, 2.0], name="a") + tf.constant([2.0, 3.0], name="b")

g = tf.Graph()
with g.device('/gpu:0'):  # 可以指定运行的方式
    result = a + b

with tf.Session() as sess:
    c = sess.run(result)
    print(c)

# 不同计算图上的张量和运算都不会共享
# 在计算图g1中定义变量V，初始化为0
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v", initializer=tf.zeros_initializer()(shape=[1]))

# 在计算图g2中定义变量V，初始化为1
g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", initializer=tf.ones_initializer()(shape=[1]))

# 在计算图g1中读取变量v的值
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

# 在计算图g2中读取变量v的值
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

# 使用会话模式有两种
# 第一种
sess = tf.Session()
# sess.run(...)
sess.close()  # 需要使用者自己关闭会话

# 第二种
# with tf.Session() as sess:
#    sess.run(...)#不需要调用Session.close()函数来关闭会话

# 通过设定默认会话计算张量的取值
sess = tf.Session()
with sess.as_default():
    print(result1.eval())

# 交互式环境构建默认会话
sess = tf.InteractiveSession()  # 构建默认会话函数
print(result1.eval())
sess.close()

# 通过ConfigProto配置会话
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)

v = tf.constant([[2.0, 2.2, 2.8], [3.5, 4.3, 6.8]])
print((tf.clip_by_value(v, 2.5, 4.5)).eval())  # tf.clip_by_value()是对取值结果进行约束的

print((tf.log(v)).eval())  # 求对数

v1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
v2 = tf.constant([[3.0, 4.0], [1.0, 2.0]])
print((v1 * v2).eval())  # 直接是对应元素相乘
print(tf.matmul(v1, v2).eval())  # 矩阵乘法

v3 = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print((tf.reduce_mean(v)).eval())  # 直接对整个矩阵求平均

# 使用了softmax回归之后的交叉熵损失函数,将softmax函数和交叉熵结合
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y,y_)

# 实现均方误差损失函数
# mes = tf.reduce_mean(tf.square(y_ - y))

# 自定义损失函数
loss = tf.reduce_sum(tf.where(tf.greater(v1,v2),(v1 - v2) * a,(v2 - v1) * b))

# tf.select 和 tf.greater 的用法
v1 = tf.constant([1.0, 2.0, 3.0, 4.0])
v2 = tf.constant([4.0, 5.0, 2.0, 7.0])

sess = tf.InteractiveSession()
print(tf.greater(v1, v2).eval())

print(tf.where(tf.greater(v1, v2), v1, v2).eval())
sess.close()

#实现自定义损失函数的应用
from numpy.random import RandomState

batch_size = 8

#两个输入节点
x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
#回归问题只有一个节点
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')

#定义一个单层的神经网络前向传播
w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w1)

#定义预测多和预测少的
loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y - y_) * loss_more,(y_ - y) * loss_less))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)

#设置回归的正确值为两个输入的和加上一个随机量
Y = [[x1 + x2 + rdm.rand()/10.0-0.05] for (x1,x2) in X]

#训练神经网络
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    step = 5000
    for i in range(step):
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size,dataset_size)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        print(sess.run(w1))


# 神经网络训练过程
batch_size = n

#每次读取一小部分数据作为当前的训练数据来执行反向传播算法
x = tf.placeholder(tf.float32,shape=(batch_size,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(batch_size,1),name='y-input')

#定义神经网络结构和优化算法
loss =
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

#训练神经网络
with tf.Session() as sess:
    #参数初始化
    #迭代的更新参数
    for i in range(steps):
        #准备好batch_size个训练数据
    current_X,current_Y =
    sess.run(train_step,feed_dict={x:current_X,y_:current_Y})

# tf.train.exponential_decay()  指数衰减学习率，当参数 staircase 是false时，为指数衰减，是true时，为阶梯状衰减
#代码实现
global_step = tf.Variable(0)

#通过exponential_decay 函数生成学习率
learning_rate = tf.train.exponential_decay(0.1,global_step,100,0.96,staircase=True)
#各个参数的意义； 1.基础学习率，2.更新次数 3.衰减速度 4.衰减率

#使用指数衰减的学习率，在minimize函数中传入global_step 将自动更新global_step 参数，从而使得学习率得到更新
learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)

#带L2正则化的损失函数
w = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w)

#loss = tf.reduce_mean(tf.square(y_ - y)) + tf.contrib.layers.l2_regularizer(lambda)(w)

weights = tf.constant([[1.0,-2.0],[-3.0,4.0]])
with tf.Session() as sess:
    print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))
    print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))

'''

'''
#通过集合计算一个5层神经网络带L2正则化的损失函数的计算方法
def get_weights(shape,lambda):

    #生成一个变量
    var = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    #将新变量的L2正则化加入集合，第一个参数是集合的名字，第二个参数是加入的内容
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lambda)(var))
    #返回生成的变量
    return var

    x = tf.placeholder(tf.float32,shape=(None,2))
    y_ = tf.placeholder(tf.float32,shape=(None,1))
    batch_size = 8
    #定义每一层网络节点的个数
    layer_dimension = [2,10,10,10,1]
    #神经网络的层数
    n_layers = len(layer_dimension)

    cur_layer = x
    #当前层的节点个数
    in_dimension = layer_dimension[0]

    #通过循环来生成5层全连接的神经网络
    for i in range(1,n_layers):
        out_dimension = layer_dimension[i]       #layer_dimension[i]为下一层节点的个数
        #生成当前层中权重的变量，将这个变量的L2正则化损失加入集合
        weight = get_weights([in_dimension,out_dimension],0.001)
        bias = tf.Variable(tf.constant(0.1,shape=[out_dimension]))

    #使用ReLU激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer,weight) + bias)
    #进入下一层之前将下一层的节点个数更新为当前节点个数
    in_dimension = layer_dimension[i]

    #损失函数
    mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

    #将均方误差损失函数加入损失集合
    tf.add_to_collection('losses',mse_loss)
    loss = tf.add_n(tf.get_collection('losses'))
'''

'''
#滑动平均模型的实现

#定义一个变量计算滑动平均
v1 = tf.Variable(0,dtype=tf.float32)

#step模拟神经网络迭代的次数，用于动态控制衰减率
step = tf.Variable(0,trainable=False)

#定义一个滑动平均的类
ema = tf.train.ExponentialMovingAverage(0.99,step)

#定义一个更新变量滑动平均的操作
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run([v1,ema.average(v1)]))

    #更新变量的值到5
    sess.run(tf.assign(v1,5))

    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))
    # 更新变量step的值到10000
    sess.run(tf.assign(step,10000))
    # 更新变量v1的值到10
    sess.run(tf.assign(v1,10))

    #更新
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))

    #再次更新
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))

v = tf.get_variable("v",shape=[1],initializer=tf.constant_initializer(1.0))
v1 = tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
print(v,v1)

with tf.variable_scope("foo"):
    v = tf.get_variable("v",[1],initializer=tf.constant_initializer(1.0))

with tf.variable_scope("foo",reuse=True):
    v1 = tf.get_variable("v",[1])
    print(v1 == v)

with tf.variable_scope("root"):
    print(tf.get_variable_scope().reuse)

    with tf.variable_scope("foo",reuse=True):
        print(tf.get_variable_scope().reuse)

        with tf.variable_scope("bar"):
            print(tf.get_variable_scope().reuse)
    print(tf.get_variable_scope().reuse)


#通过tf.variable_scope()管理变量名称
v1 = tf.get_variable("v",[1])
print(v1.name)

with tf.variable_scope("foo"):
    v2 = tf.get_variable("v",[1])
    print(v2.name)

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v",[1])
        print(v3.name)

    v4 = tf.get_variable("v1",[1])
    print(v4.name)

with tf.variable_scope("",reuse=True):
    v5 = tf.get_variable("foo/bar/v",[1])
    print(v5 == v3)

    v6 = tf.get_variable("foo/v1",[1])
    print(v6 == v4)
'''
'''
# 持久化代码实现
v1 = tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
v2 = tf.Variable(tf.constant(2.0,shape=[1]),name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()

#声明tf,train.Saver类用于保存模型
saver = tf.train.Saver({"v1":v1,"v2":v2})

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess,"E:\MachineLearning/model/model.ckpt")


#加载已经保存的模型
v3 = tf.Variable(tf.constant(1.0,shape=[1]),name="other-v1")
v4 = tf.Variable(tf.constant(2.0,shape=[1]),name="other-v2")

saver = tf.train.import_meta_graph("E:\MachineLearning\model\model.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess,"E:\MachineLearning\model\model.ckpt")
    #通过张量的名称来获取张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))


v = tf.Variable(0,dtype=tf.float32,name="v")
for variables in tf.global_variables():
    print(variables.name)

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())

for variables in tf.global_variables():
    print(variables.name)

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    sess.run(tf.assign(v,10))
    sess.run(maintain_averages_op)

    saver.save(sess,"E:\MachineLearning\model\model1.ckpt")
    print(sess.run([v,ema.average(v)]))

#通过变量重命名将v的滑动平均直接赋值给v
v = tf.Variable(0,dtype=tf.float32,name='v')
saver = tf.train.Saver({"v/ExponentialMovingAverage":v})
with tf.Session() as sess:
    saver.restore(sess,"E:\MachineLearning\model\model1.ckpt")
    print(sess.run(v))


#使用variables_to_restore()函数直接生成上述代码提供的字典
v = tf.Variable(0,dtype=tf.float32,name="v")
ema = tf.train.ExponentialMovingAverage(0.99)
print(ema.variables_to_restore())
saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess,"E:\MachineLearning\model\model1.ckpt")
    print(sess.run(v))


#通过convert_variables_to_constants函数可以将图中的变量通过常量形式保存，整个计算图可以统一存放在一个文件中
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
v2 = tf.Variable(tf.constant(2.0,shape=[1]),name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    #导出当前计算图
    graph_def = tf.get_default_graph().as_graph_def()

    #将其中的变量转换为常量进行保存
    output_graph_def = graph_util.convert_variables_to_constants(sess,graph_def,['add'])

    #将导出的模型存入文件
    with tf.gfile.GFile("E:\MachineLearning\model\combined-model.pb","wb") as f:
        f.write(output_graph_def.SerializeToString())
        
'''
'''
#直接计算定义的加法运算的结果
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename = "E:\MachineLearning\model\combined-model.pb"
    #读取保存的文件
    with gfile.FastGFile(model_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        #加载图
        result = tf.import_graph_def(graph_def,return_elements=["add:0"])
        print(sess.run(result))

#查看计算图的元图
v1 = tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
v2 = tf.Variable(tf.constant(2.0,shape=[1]),name="v2")
result = v1 + v2
saver = tf.train.Saver()
saver.export_meta_graph("E:\MachineLearning\model\model.ckpt.meda.json",as_text=True)


#查看model.ckpt中保存的变量信息
reader = tf.train.NewCheckpointReader('E:\MachineLearning\model\model.ckpt')

#获取所有变量列表
all_variables = reader.get_variable_to_shape_map()
for variable_name in all_variables:
    print(variable_name,all_variables[variable_name])

#获取名称为v1的变量的取值
print("Value for variable v1 is " , reader.get_tensor("v1"))

'''

'''
#实现一个卷积层的前向传播过程
filter_weight = tf.get_variable('weights',[5,5,3,16])#参数说明：1.2 ：过滤器大小 3：当前层的深度 4：过滤器深度
initializer = tf.truncated_normal_initializer(stddev=0.1)

biases = tf.get_variable('biases',[16],initializer=tf.constant_initializer(0.1))

#tf.nn.conv2d参数：1.对应一个batch 2.卷积层的权重 3.步长 4.填充的方法，SAME全0填充，VALID不添加
conv = tf.nn.conv2d(input,filter_weight,strides=[1,1,1,1],padding='SAME')

#给每一个节点添加偏置项
bias = tf.nn.bias_add(conv,biases)

#通过relu函数去线性化
actived_conv = tf.nn.relu(bias)


#实现最大池化层的前向传播
pool = tf.nn.max_pool(actived_conv,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')#ksize过滤器尺寸，strides步长大小，padding是否全0填充
'''

#直接使用TensorFlow原始API实现卷积层
with tf.Session() as sess:
    weights = tf.get_variable("weight",...)
    biases = tf.get_variable("bias",...)
    conv = tf.nn.conv2d(...)

relu = tf.nn.relu(tf.nn.bias_add(conv,biases))

#使用TensorFlow-Slim实现卷积层
net = slim.conv2d(input,32,[3,3])#第一个参数是输入节点矩阵，第二个参数是过滤器深度，第三个参数是过滤器尺寸

#实现Inception-v3模型中的Inception模块
with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='SAME'):
    net = 上一层的输出节点矩阵
    with tf.variable_scope('Mixed_7c'):
        with tf.variable_scope('Branch_0'):
            #实现一个过滤器边长1，深度320的卷积层
            branch_0 = slim.conv2d(net,320,[1,1],scope='Conv2d_0a_1x1')

        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net,384,[1,1],scope='Conv2d_0a_1x1')
            branch_1 = tf.concat(3,[slim.conv2d(branch_1,384,[1,3],scope='Conv2d_0b_1x3'),slim.conv2d(branch_1,384,[3,1],scope='Conv2d_0c_3x1')])# 3代表的是拼接的维度

        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net,448,[1,1],scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2,384,[3,3],scope='Conv2d_0b_3x3')
            branch_2 = tf.concat(3,[slim.conv2d(branch_2,384,[1,3],scope='Conv2d_0c_1x3'),slim.conv2d(branch_2,384,[3,1],scope='Conv2d_0d_3x1')])


        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3,192,[1,1],scope='Conv2d_0b_1x1')

        #输出是上面的4个结果拼接得到的
        net = tf.concat(3,[branch_0,branch_1,branch_2,branch_3])

