import tensorflow as tf
import numpy as np
from cifar10 import cifar10,cifar10_input
import time

'''
初始化权重函数
'''
def variable_with_weight_loss(shape,std,w1):
    var = tf.Variable(tf.truncated_normal(shape,stddev=std),dtype=tf.float32)
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var),w1,name="weight_loss")
        tf.add_to_collection("losses",weight_loss)
    return var

'''
损失函数
'''
def loss_func(logits,labels):
    labels = tf.cast(labels,tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                           labels=labels,name="cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(tf.reduce_sum(cross_entropy))
    tf.add_to_collection("losses",cross_entropy_mean)
    return tf.add_n(tf.get_collection("losses"),name="total_loss")


if __name__ == "__main__":
    #设置最大迭代次数
    max_steps = 10000
    #设置每次训练的数据大小
    batch_size = 128
    #下载解压数据
    cifar10.maybe_download_and_extract()
    # 设置数据的存放目录
    cifar10_dir = "D:/dataset/cifar10_data/cifar-10-batches-bin"
    #获取数据增强后的训练集数据
    images_train,labels_train = cifar10_input.distorted_inputs(cifar10_dir,batch_size)
    #获取裁剪后的测试数据
    images_test,labels_test = cifar10_input.inputs(eval_data=True,data_dir=cifar10_dir
                                                   ,batch_size=batch_size)
    #定义模型的输入和输出数据
    image_holder = tf.placeholder(dtype=tf.float32,shape=[batch_size,24,24,3])
    label_holder = tf.placeholder(dtype=tf.int32,shape=[batch_size])

    #设计第一层卷积
    weight1 = variable_with_weight_loss(shape=[5,5,3,64],std=5e-2,w1=0)
    kernel1 = tf.nn.conv2d(image_holder,weight1,[1,1,1,1],padding="SAME")
    bais1 = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[64]))
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1,bais1))
    pool1 = tf.nn.max_pool(conv1,[1,3,3,1],[1,2,2,1],padding="SAME")
    norm1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001 / 9,beta=0.75)

    #设计第二层卷积
    weight2 = variable_with_weight_loss(shape=[5,5,64,64],std=5e-2,w1=0)
    kernel2 = tf.nn.conv2d(norm1,weight2,[1,1,1,1],padding="SAME")
    bais2 = tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(kernel2,bais2))
    norm2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.01 / 9,beta=0.75)
    pool2 = tf.nn.max_pool(norm2,[1,3,3,1],[1,2,2,1],padding="SAME")

    #第一层全连接层
    reshape = tf.reshape(pool2,[batch_size,-1])
    dim = reshape.get_shape()[1].value
    weight3 = variable_with_weight_loss([dim,384],std=0.04,w1=0.004)
    bais3 = tf.Variable(tf.constant(0.1,shape=[384],dtype=tf.float32))
    local3 = tf.nn.relu(tf.matmul(reshape,weight3)+bais3)

    #第二层全连接层
    weight4 = variable_with_weight_loss([384,192],std=0.04,w1=0.004)
    bais4 = tf.Variable(tf.constant(0.1,shape=[192],dtype=tf.float32))
    local4 = tf.nn.relu(tf.matmul(local3,weight4)+bais4)

    #最后一层
    weight5 = variable_with_weight_loss([192,10],std=1/192.0,w1=0)
    bais5 = tf.Variable(tf.constant(0.0,shape=[10],dtype=tf.float32))
    logits = tf.add(tf.matmul(local4,weight5),bais5)

    #获取损失函数
    loss = loss_func(logits,label_holder)
    #设置优化算法使得成本最小
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
    #获取最高类的分类准确率，取top1作为衡量标准
    top_k_op = tf.nn.in_top_k(logits,label_holder,1)
    #创建会话
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    #启动图片数据增强队列
    tf.train.start_queue_runners()
    #开始训练
    for step in range(max_steps):
        start_time = time.time()
        images_batch,labels_batch = sess.run([images_train,labels_train])
        _,loss_value = sess.run([train_step,loss],feed_dict={image_holder:images_batch,
                                                             label_holder:labels_batch})
        #获取计算时间
        duration = time.time() - start_time
        if step % 1000 == 0:
            #计算每秒处理多少张图片
            per_images_second = batch_size / duration
            #获取时间
            sec_per_batch = float(duration)
            print("step:%d,duration:%.3f,per_images_second:%.2f,loss:%.3f"%(step,duration
                                                                ,per_images_second,loss_value))

    #计算测试集上的准确率
    num_examples = 10000
    import math
    num_iter = int(math.ceil(num_examples / batch_size))
    true_count = 0
    total_sample_count = num_iter * batch_size
    step = 0
    while step < num_iter:
        images_batch,labels_batch = sess.run([images_test,labels_test])
        pred = sess.run([top_k_op],feed_dict={image_holder:images_batch,label_holder:labels_batch})
        true_count += np.sum(pred)
        step += 1
    #计算测试集的准确率
    precision = true_count / total_sample_count
    print("test accuracy:%.3f"%precision)
    #test accuracy:0.805





















