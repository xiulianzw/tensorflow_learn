import tensorflow as tf
from tensorflow.examples.tutorials import mnist

'''
循环神经网络参数设置
'''
class rnn_config(object):
    #学习率设置
    learning_rate = 0.001
    #训练迭代次数
    training_iters = 10000
    #batch size设置
    batch_size = 128
    #输入层的大小
    n_inputs = 28
    #RNN的长度
    n_steps = 28
    #隐藏层的神经元个数
    n_hidden_units = 128
    #输出类别的数量,0-9一共10个数字
    n_classes = 10

'''
循环神经网络的结构
'''
class RNN_MNIST(object):
    def __init__(self,config):
        self.config = config
        self.x = tf.placeholder(tf.float32,[None,config.n_steps,config.n_inputs])
        self.y = tf.placeholder(tf.float32,[None,config.n_classes])
        #定义权重
        self.weights = {
            #输入层的权重 (28,128)
            "in":tf.Variable(tf.random_normal([self.config.n_inputs,self.config.n_hidden_units])),
            #输出层权重 (128,10)
            "out":tf.Variable(tf.random_normal([self.config.n_hidden_units,self.config.n_classes]))
        }
        #定义偏置
        self.biases = {
            #输入层的偏置 (128,)
            "in":tf.Variable(tf.constant(0.1,shape=[self.config.n_hidden_units,])),
            #输出层的偏置(10,)
            "out":tf.Variable(tf.constant(0.1,shape=[self.config.n_classes,]))
        }
        self.rnn()

    def rnn(self):
        #隐藏层
        X = tf.reshape(self.x,[-1,self.config.n_inputs])
        X_in = tf.matmul(X,self.weights["in"]) + self.biases["in"]
        X_in = tf.reshape(X_in,[-1,self.config.n_steps,self.config.n_hidden_units])
        #lstm核
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden_units,
                                                 forget_bias=1.0,state_is_tuple=True)
        #初始化
        init_state = lstm_cell.zero_state(self.config.batch_size,dtype=tf.float32)
        outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=init_state,
                                                time_major=False)
        #预测类标
        self.y_pred = tf.matmul(final_state[1],self.weights["out"])+self.biases["out"]
        #计算交叉熵
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred,labels=self.y)
        #计算损失函数
        self.loss = tf.reduce_mean(cross_entropy)
        #使用Adam最小化损失函数
        self.Adam = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        #计算模型的准确率
        correct_pred = tf.equal(tf.argmax(self.y_pred,1),tf.argmax(self.y,1))
        self.acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))


def training():
    #获取数据集
    mnist_data = mnist.input_data.read_data_sets("/MNIST_data",one_hot=True)
    #初始化循环神经网络的配置参数
    config = rnn_config()
    #初始化模型
    rnn_model = RNN_MNIST(config)
    with tf.Session() as sess:
        #初始化所有变量
        sess.run(tf.global_variables_initializer())
        #训练
        for step in range(config.training_iters):
            batch_xs,batch_ys = mnist_data.train.next_batch(config.batch_size)
            batch_xs = batch_xs.reshape([config.batch_size,config.n_steps,config.n_inputs])
            #运行优化器,最小化损失函数
            sess.run([rnn_model.Adam],feed_dict={rnn_model.x:batch_xs,rnn_model.y:batch_ys})
            if step % 1000 == 0:
                loss,acc = sess.run([rnn_model.loss,rnn_model.acc],
                                feed_dict={rnn_model.x:batch_xs,rnn_model.y:batch_ys})
                print("step:%d,loss:%.3f,train accuracy:%.3f"%(step,loss,acc))
        #计算模型在测试集上的准确率和损失值
        test_loss_total = 0
        test_acc_total = 0
        test_num = 0
        for i in range(100):
            test_x,test_y = mnist_data.test.next_batch(config.batch_size)
            test_x = test_x.reshape([config.batch_size,config.n_steps,config.n_inputs])
            test_loss,test_acc = sess.run([rnn_model.loss,rnn_model.acc],
                                          feed_dict={rnn_model.x:test_x,rnn_model.y:test_y})
            test_loss_total += test_loss * len(test_x)
            test_acc_total += test_acc * len(test_x)
            test_num += len(test_x)
        print("test loss:%.3f,test accuracy acc:%.3f"%(test_loss_total/test_num,test_acc_total/test_num))


if __name__ == "__main__":
    training()
