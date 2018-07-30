# 定义神经网络的前向传播过程以及参数
import tensorflow as tf

# 定义神经网络结构相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SEZE = 5

# 第一层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SEZE = 5

# 全连接层的节点个数
FC_SIZE = 512

'''
#通过tf.get_variable函数获取变量
def get_weight_variable(shape,regularizer):
    weights = tf.get_variable("weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1))

    #如果给出正则化，加入到损失函数中
    if(regularizer != None):
        tf.add_to_collection('losses',regularizer(weights))
    return weights

#定义神经网络前向传播
def inference(input_tensor,regularizer):
    #声明第一层神经网络的变量并完成前向传播
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases = tf.get_variable("biases",[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights) + biases)

        #声明第二层
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases = tf.get_variable("biases",[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1,weights) + biases

    return layer2

'''


def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weights', [CONV1_SEZE, CONV1_SEZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        # 使用5x5，深度为32的过滤器，步长为1，全0填充, 28x28x1
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

        # 第一层池化层，28x28x32
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 第二层卷积
    with tf.variable_scope('layer3-conv2'):  # 14x14x64
        conv2_weights = tf.get_variable('weights', [CONV2_SEZE, CONV2_SEZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

        # 第二层池化层,7x7x64
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

   #全连接层，将7x7x64的矩阵格式为向量
    pool_shape = pool2.get_shape().as_list()

    #计算格式为向量后的长度
    node = pool_shape[1] * pool_shape[2] * pool_shape[3]

    #将第四层的输出变为一个batch的向量
    reshape = tf.reshape(pool2,[pool_shape[0],node])

    #声明第五层
    with tf.variable_scope('layer5-fc1'):
        fc1_weights= tf.get_variable('weight',[node,FC_SIZE],initializer=tf.initialize_all_variables(stddev=0.1))

        #只有全连接层的权重需加入正则化
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias',[FC_SIZE],initializer=tf.initialize_all_variables(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshape,fc1_weights) + fc1_biases)
        if train : fc1 = tf.nn.dropout(fc1,0.5)

    #声明第六层
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weight',[FC_SIZE,NUM_LABELS],initializer=tf.initialize_all_variables(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias',[NUM_LABELS],initializer=tf.initialize_all_variables(0.1))
        logit = tf.matmul(fc1,fc2_weights) + fc2_biases

    return logit