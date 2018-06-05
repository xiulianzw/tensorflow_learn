from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

if __name__ == "__main__":
    #加载数据
    mnist_data = input_data.read_data_sets("MNIST_data/",one_hot=True)
    #获取训练数据
    train_data = mnist_data.train
    #获取验证数据
    validation_data = mnist_data.validation
    #获取测试数据
    test_data = mnist_data.test
    #定义输入变量X
    X = tf.placeholder(tf.float32,[None,784])
    #定义权重
    w = tf.Variable(tf.zeros([784,10],dtype=tf.float32))
    #定义偏置
    b = tf.Variable(tf.zeros([10]),dtype=tf.float32)
    #计算输出y
    Y = tf.nn.softmax(tf.matmul(X,w)+b)
    #定义预测时的输出
    Y_ = tf.placeholder(tf.float32,[None,10])
    #定义损失函数
    loss_func = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y),reduction_indices=[1]))
    #定义训练时候的优化函数
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss_func)
    #定义一个交互式的会话
    sess = tf.InteractiveSession()
    #初始化变量
    tf.global_variables_initializer().run()
    #开始迭代
    for i in range(100000):
        #每次获取一小块的数据
        batch_xs,batch_ys = train_data.next_batch(100)
        train_step.run({X:batch_xs,Y_:batch_ys})
    #计算准确率
    correct_pred = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    #计算模型在训练集上的准确率
    print("train accray:%.4f"%accuracy.eval({X:train_data.images,Y_:train_data.labels}))
    #计算模型在测试集上的准确率
    print("test accuracy:%.4f"%accuracy.eval({X:test_data.images,Y_:test_data.labels}))
    '''
    train accray:0.9257
    test accuracy:0.9221
    '''
