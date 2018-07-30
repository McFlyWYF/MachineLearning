import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#加载mnist_inference.py 和 mnist_train.py 中定义的常量和函数
import mnist_inference
import mnist_train

#每10秒加载一次最新的模型，在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:

        #定义输入输出的格式
        x = tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name='x-input')
        y_ = tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name='y-input')
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}

        #直接使用封装好的函数计算前向传播结果
        y = mnist_inference.inference(x,None)

        #使用前向传播结果计算正确率
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        #通过变量重命名的方式加载模型，不需要调用求滑动平均的函数来求平均值
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        #每隔10秒调用一次计算正确率的过程
        while True:
            with tf.Session() as sess:
            #找到最新的模型文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if(ckpt and ckpt.model_checkpoint_path):
                    #加载模型
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    #通过文件名得到模型保存时迭代的次数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('/')[-1]
                    accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
                    print('After %s training step(s),validation ' " accuracy = %g" % (global_step,accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv = None):
    mnist = input_data.read_data_sets("/tmp/data",one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()