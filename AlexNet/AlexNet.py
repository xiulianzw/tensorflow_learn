from datetime import datetime
import math,time
import tensorflow as tf

batch_size = 32
num_bathes = 100

'''
获取tensor信息
'''
def print_tensor_info(tensor):
    print("tensor name:",tensor.op.name,"-tensor shape:",tensor.get_shape().as_list())

'''
计算每次迭代消耗时间
session:TensorFlow的Session
target:需要评测的运算算子
info_string:测试的名称
'''
def time_tensorflow_run(session,target,info_string):
    #前10次迭代不计入时间消耗
    num_step_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_bathes + num_step_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_step_burn_in:
            if not i % 10 :
                print("%s:step %d,duration=%.3f"%(datetime.now(),i-num_step_burn_in,duration))
            total_duration += duration
            total_duration_squared += duration * duration
    #计算消耗时间的平均差
    mn = total_duration / num_bathes
    #计算消耗时间的标准差
    vr = total_duration_squared / num_bathes - mn * mn
    std = math.sqrt(vr)
    print("%s:%s across %d steps,%.3f +/- %.3f sec / batch"%(datetime.now(),info_string,num_bathes,
                                                             mn,std))
#主函数
def run_bechmark():
    with tf.Graph().as_default():
        image_size = 224
        #以高斯分布产生一些图片
        images = tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],
                                              dtype=tf.float32,stddev=0.1))
        output,parameters = inference(images)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        time_tensorflow_run(sess,output,"Forward")
        objective = tf.nn.l2_loss(output)
        grad = tf.gradients(objective,parameters)
        time_tensorflow_run(sess,grad,"Forward-backward")


def inference(images):

    #定义参数
    parameters = []

    #第一层卷积层
    with tf.name_scope("conv1") as scope:
        #设置卷积核11×11,3通道,64个卷积核
        kernel1 = tf.Variable(tf.truncated_normal([11,11,3,64],mean=0,stddev=0.1,
                                                  dtype=tf.float32),name="weights")
        #卷积,卷积的横向步长和竖向补偿都为4
        conv = tf.nn.conv2d(images,kernel1,[1,4,4,1],padding="SAME")
        #初始化偏置
        biases = tf.Variable(tf.constant(0,shape=[64],dtype=tf.float32),trainable=True,name="biases")
        bias = tf.nn.bias_add(conv,biases)
        #RELU激活函数
        conv1 = tf.nn.relu(bias,name=scope)
        #输出该层的信息
        print_tensor_info(conv1)
        #统计参数
        parameters += [kernel1,biases]
        #lrn处理
        lrn1 = tf.nn.lrn(conv1,4,bias=1,alpha=1e-3/9,beta=0.75,name="lrn1")
        #最大池化
        pool1 = tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID",name="pool1")
        print_tensor_info(pool1)

    #第二层卷积层
    with tf.name_scope("conv2") as scope:
        #初始化权重
        kernel2 = tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,stddev=0.1)
                              ,name="weights")
        conv = tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding="SAME")
        #初始化偏置
        biases = tf.Variable(tf.constant(0,dtype=tf.float32,shape=[192])
                             ,trainable=True,name="biases")
        bias = tf.nn.bias_add(conv,biases)
        #RELU激活
        conv2 = tf.nn.relu(bias,name=scope)
        print_tensor_info(conv2)
        parameters += [kernel2,biases]
        #LRN
        lrn2 = tf.nn.lrn(conv2,4,1.0,alpha=1e-3/9,beta=0.75,name="lrn2")
        #最大池化
        pool2 = tf.nn.max_pool(lrn2,[1,3,3,1],[1,2,2,1],padding="VALID",name="pool2")
        print_tensor_info(pool2)

    #第三层卷积层
    with tf.name_scope("conv3") as scope:
        #初始化权重
        kernel3 = tf.Variable(tf.truncated_normal([3,3,192,384],dtype=tf.float32,stddev=0.1)
                              ,name="weights")
        conv = tf.nn.conv2d(pool2,kernel3,strides=[1,1,1,1],padding="SAME")
        biases = tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),trainable=True,name="biases")
        bias = tf.nn.bias_add(conv,biases)
        #RELU激活层
        conv3 = tf.nn.relu(bias,name=scope)
        parameters += [kernel3,biases]
        print_tensor_info(conv3)

    #第四层卷积层
    with tf.name_scope("conv4") as scope:
        #初始化权重
        kernel4 = tf.Variable(tf.truncated_normal([3,3,384,256],stddev=0.1,dtype=tf.float32),
                              name="weights")
        #卷积
        conv = tf.nn.conv2d(conv3,kernel4,strides=[1,1,1,1],padding="SAME")
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[256]),trainable=True,name="biases")
        bias = tf.nn.bias_add(conv,biases)
        #RELU激活
        conv4 = tf.nn.relu(bias,name=scope)
        parameters += [kernel4,biases]
        print_tensor_info(conv4)

    #第五层卷积层
    with tf.name_scope("conv5") as scope:
        #初始化权重
        kernel5 = tf.Variable(tf.truncated_normal([3,3,256,256],stddev=0.1,dtype=tf.float32),
                              name="weights")
        conv = tf.nn.conv2d(conv4,kernel5,strides=[1,1,1,1],padding="SAME")
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[256]),name="biases")
        bias = tf.nn.bias_add(conv,biases)
        #REUL激活层
        conv5 = tf.nn.relu(bias)
        parameters += [kernel5,bias]
        #最大池化
        pool5 = tf.nn.max_pool(conv5,[1,3,3,1],[1,2,2,1],padding="VALID",name="pool5")
        print_tensor_info(pool5)

    #第六层全连接层
    pool5 = tf.reshape(pool5,(-1,6*6*256))
    weight6 = tf.Variable(tf.truncated_normal([6*6*256,4096],stddev=0.1,dtype=tf.float32),
                           name="weight6")
    ful_bias1 = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[4096]),name="ful_bias1")
    ful_con1 = tf.nn.relu(tf.add(tf.matmul(pool5,weight6),ful_bias1))

    #第七层第二层全连接层
    weight7 = tf.Variable(tf.truncated_normal([4096,4096],stddev=0.1,dtype=tf.float32),
                          name="weight7")
    ful_bias2 = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[4096]),name="ful_bias2")
    ful_con2 = tf.nn.relu(tf.add(tf.matmul(ful_con1,weight7),ful_bias2))
    #
    #第八层第三层全连接层
    weight8 = tf.Variable(tf.truncated_normal([4096,1000],stddev=0.1,dtype=tf.float32),
                          name="weight8")
    ful_bias3 = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[1000]),name="ful_bias3")
    ful_con3 = tf.nn.relu(tf.add(tf.matmul(ful_con2,weight8),ful_bias3))

    #softmax层
    weight9 = tf.Variable(tf.truncated_normal([1000,10],stddev=0.1),dtype=tf.float32,name="weight9")
    bias9 = tf.Variable(tf.constant(0.0,shape=[10]),dtype=tf.float32,name="bias9")
    output_softmax = tf.nn.softmax(tf.matmul(ful_con3,weight9)+bias9)

    return output_softmax,parameters


if __name__ == "__main__":
    run_bechmark()
