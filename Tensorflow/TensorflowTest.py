import tensorflow as tf
import numpy as np

x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.100, 0.200], x_data) + 0.300

b = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
y = tf.matmul(w,x_data) + b


#最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#初始化变量
init = tf.initialize_all_variables()

#启动图
sess = tf.Session()
sess.run(init)

#拟合平面
for step in range(0,201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(w),sess.run(b))

#使用feed和fetch为任意的操作赋值或者获取数据

#创建图
matrix1 = tf.constant([[3.,3.]])#1x2矩阵
matrix2 = tf.constant([[2.],[2.]])#2x1矩阵

product = tf.matmul(matrix1,matrix2)#矩阵相乘
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()#显式关闭会话

#自动关闭会话
with tf.Session() as sess:
    with tf.device("/cpu:0"):#指定使用机器的CPU
        result = sess.run(product)
        print(result)

#交互式会话
sess = tf.InteractiveSession()
x = tf.Variable([1.0,2.0])
a = tf.constant([3.0,3.0])

x.initializer.run()

sub = tf.subtract(x,a)#减法，x - a
print(sub.eval())

#张量
#变量
state = tf.Variable(0,name = "counter")#初始化为0
one = tf.constant(1)
new_value = tf.add(state,one)#使state加1
update = tf.assign(state,new_value)#更新

init_op = tf.global_variables_initializer()#增加初始化op

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))

    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


#Fetch，获取值
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2,input3)
mul = tf.multiply(input1,intermed)#相乘

with tf.Session() as sess:
    result = sess.run([mul,intermed])
    print(result)

#Feed, feed 只在调用它的方法内有效, 方法结束, feed 就会消失,标记的方法是使用 tf.placeholder() 为这些操作创建占位符.
input1 = tf.placeholder(tf.types.float32)
input2 = tf.placeholder(tf.types.float32)
output = tf.multiply(input1,input2)

with tf.Session() as tf:
    print(sess.run([output],feed_dict={input1:[7.],input2:[2.]}))


a = tf.placeholder("float")
b = tf.placeholder("float")

y = tf.multiply(a,b)#构造一个op节点

sess = tf.Session()
print(sess.run(y,feed_dict={a:3,b:3}))
sess.close()

a = tf.constant(3)
b = tf.constant(2)
c = tf.constant(-1)
d = tf.constant([1.2,4.3,2.9])

tf.add(a,b)#求和
tf.subtract(a,b)#减法
tf.multiply(a,b)#乘法
tf.div(a,b)#除法
tf.mod(a,b)#取模
tf.abs(c)#求绝对值
tf.negative(a)#取负
tf.sign(a)#返回符号
tf.inv(a)#取反
tf.square(a)#计算平方
tf.round(d)#舍入最接近的整数
tf.sqrt(a)#开方
tf.pow(a,b)#a的b次方
tf.exp(a)#e的a次方
tf.log(a)#一次输入是以e为底a的对数，两次输入是以第二个为底
tf.maximum(a,b)#返回最大值
tf.minimum(a,b)#返回最小值
tf.cos(a)#三角函数cos



#数据类型转换
e = tf.constant("abcde")
tf.string_to_number(e)#字符串转换为数字
tf.to_double(a)
tf.cast(a)#转换为整数，比如1.8 = 1,2.2 = 2

#形状操作
tf.shape()#返回数据的shape
tf.size()#返回数据的元素数量
tf.rank()#返回tensor的rank
tf.reshape()#改变tensor的形状
tf.expand_dims()#插入维度1进入一个tensor

#切片与合并
tf.slice()#切片操作
tf.split()#沿着某一维度将tensor分离
tf.concat()#沿着某一维度连接
tf.pack()#将一系列rank-R的tensor打包为一个rank-(R+1)的tensor
tf.reverse()#沿着某维度进行序列反转
tf.transpose()#调换tensor的维度顺序
tf.gather()#合并索引所指示的切片

