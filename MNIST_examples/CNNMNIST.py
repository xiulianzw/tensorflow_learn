import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
初始化权重
'''
def initial_weights(weight_shape):
    weights = tf.truncated_normal(weight_shape,mean=0.0,stddev=0.1,dtype=tf.float32)
    return tf.Variable(weights)

'''
初始化截距
'''
def initial_bais(bais_shape):
    bais = tf.constant(0.1,shape=bais_shape)
    return tf.Variable(bais)

'''
定义卷积函数
'''
def conv2d(X,w):
    return tf.nn.conv2d(X,w,strides=[1,1,1,1],padding="SAME")

'''
定义池化函数
'''
def max_pool(X):
    return tf.nn.max_pool(X,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

if __name__ == "__main__":
    #获取数据
    mnist_data = input_data.read_data_sets("/MNIST_data",one_hot=True)
    #创建一个会话
    sess = tf.InteractiveSession()
    #定义输入和输出
    x = tf.placeholder(dtype=tf.float32,shape=[None,784])
    x_image = tf.reshape(x,[-1,28,28,1])
    y_ = tf.placeholder(dtype=tf.float32,shape=[None,10])
    #设计前向传播网络
    #第一层卷积
    w_conv1 = initial_weights([5,5,1,32])
    b_conv1 = initial_bais([32])
    h_conv1 = tf.nn.relu(tf.add(conv2d(x_image,w_conv1),b_conv1))
    h_pool1 = max_pool(h_conv1)
    #第二层卷积
    w_conv2 = initial_weights([5,5,32,64])
    b_conv2 = initial_bais([64])
    h_conv2 = tf.nn.relu(tf.add(conv2d(h_pool1,w_conv2),b_conv2))
    h_pool2 = max_pool(h_conv2)
    #全连接
    w_fc1 = initial_weights([7*7*64,1024])
    b_fc1 = initial_bais([1024])
    h_pool2_flat = tf.reshape(h_pool2,shape=[-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
    #dropout防止过拟合
    keep_prob = tf.placeholder(dtype=tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
    #softmax层
    w_fc2 = initial_weights([1024,10])
    b_fc2 = initial_bais([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)
    #定义损失函数
    loss_func = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_func)
    #计算模型的准确率
    correct_pred = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    #迭代
    tf.global_variables_initializer().run()
    for i in range(20000):
        batch_xs,batch_ys = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        if i % 1000 == 0:
            #评估模型在训练集上的准确率
            train_accuracy = accuracy.eval({x:batch_xs,y_:batch_ys,keep_prob:1.0})
            print("step:",i,"-train accuracy:%.4f"%train_accuracy)
    #模型在测试集上的准确率
    print("test accuracy:%.4f"%(accuracy.eval({x:mnist_data.test.images,y_:mnist_data.test.labels,keep_prob:1.0})))
    #test accuracy:0.9920
