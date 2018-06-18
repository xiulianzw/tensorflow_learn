from datetime import datetime
import math,time
import tensorflow as tf

'''
定义卷积层函数
input_op:输入的tensor
name：该层的名称
kh:卷积核的高
kw:卷积核的宽
n_out:卷积核的数量(输出通道数)
dh:步长的高
dw:步长的宽
p:参数列表
'''
def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        #初始化权重
        kernel = tf.get_variable(scope+"w",shape=[kh,kw,n_in,n_out],dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        #卷积
        conv = tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding="SAME")
        #初始化偏置
        bias_init_val = tf.constant(0.0,shape=[n_out],dtype=tf.float32)
        biases = tf.Variable(bias_init_val,trainable=True,name="b")
        z = tf.nn.bias_add(conv,biases)
        activation = tf.nn.relu(z,name=scope)
        #保存参数
        p += [kernel,biases]
    return activation

'''
定义全连接层函数
input_op:输入的tensor
name:该层的名称
n_out:输出的通道数
p:参数列表
'''
def fc_op(input_op,name,n_out,p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        #初始化全连接的权重
        kernel = tf.get_variable(scope+"w",shape=[n_in,n_out],dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        #初始化全连接层的偏置
        biases = tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name="b")
        #将输入与权重的乘法和偏置的加法合并
        activation = tf.nn.relu_layer(input_op,kernel,biases,name=scope)
        #保存参数
        p += [kernel,biases]
        return activation

'''
定义最大池化层
input_op:输入的tensor
name:该层的名称
kh:池化层的高
kw:池化层的宽
dh:步长的高
dw:步长的宽
'''
def max_pool(input_op,name,kh,kw,dh,dw):
    return tf.nn.max_pool(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1]
                          ,padding="SAME",name=name)

'''
VGG16
'''
def inference_op(input_op,keep_prob):
    p = []
    #第一层的第一层卷积
    conv1_1 = conv_op(input_op,name="conv1_1",kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
    #第一层的第二层卷积
    conv1_2 = conv_op(conv1_1,name="conv1_2",kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
    #最大池化层
    pool1 = max_pool(conv1_2,name="pool1",kh=2,kw=2,dw=2,dh=2)

    #第二层的第一层卷积
    conv2_1 = conv_op(pool1,name="conv2_1",kh=3,kw=3,n_out=128,dh=1,dw=1,p=p)
    #第二层的第二层卷积
    conv2_2 = conv_op(conv2_1,name="conv2_2",kh=3,kw=3,n_out=128,dh=1,dw=1,p=p)
    #第二层的最大池化
    pool2 = max_pool(conv2_2,name="pool2",kh=2,kw=2,dh=2,dw=2)

    #第三层
    conv3_1 = conv_op(pool2,name="conv3_1",kh=3,kw=3,n_out=256,dh=1,dw=1,p=p)
    conv3_2 = conv_op(conv3_1,name="conv3_2",kh=3,kw=3,n_out=256,dh=1,dw=1,p=p)
    conv3_3 = conv_op(conv3_2,name="conv3_3",kh=3,kw=3,n_out=256,dh=1,dw=1,p=p)
    pool3 = max_pool(conv3_3,name="pool3",kh=2,kw=2,dh=2,dw=2)

    #第四层
    conv4_1 = conv_op(pool3,name="conv4_1",kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)
    conv4_2 = conv_op(conv4_1,name="conv4_2",kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)
    conv4_3 = conv_op(conv4_2,name="conv4_3",kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)
    pool4 = max_pool(conv4_3,name="pool4",kh=2,kw=2,dh=2,dw=2)

    #第五层
    conv5_1 = conv_op(pool4,name="conv5_1",kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)
    conv5_2 = conv_op(conv5_1,name="conv5_2",kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)
    conv5_3 = conv_op(conv5_2,name="conv5_3",kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)
    pool5 = max_pool(conv5_3,name="pool5",kh=2,kw=2,dh=2,dw=2)
    #将pool5展平
    pool5_shape = pool5.get_shape()
    flattened_shape = pool5_shape[1].value * pool5_shape[2].value * pool5_shape[3].value
    resh1 = tf.reshape(pool5,[-1,flattened_shape],name="resh1")

    #全连接层
    fc6 = fc_op(resh1,name="fc6",n_out=4096,p=p)
    fc6_drop = tf.nn.dropout(fc6,keep_prob,name="fc6_drop")

    #全连接层
    fc7 = fc_op(fc6_drop,name="fc7",n_out=4096,p=p)
    fc7_drop = tf.nn.dropout(fc7,keep_prob,name="fc7_drop")

    fc8 = fc_op(fc7_drop,name="fc8",n_out=1000,p=p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax,1)
    return predictions,softmax,fc8,p


num_batches = 100
def time_tensorflow_run(session,target,feed,info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _= session.run(target,feed_dict=feed)
        duration = time.time() - start_time
        if i > num_steps_burn_in:
            if not i % 10:
                print("%s：step:%d,duration:%.3f"%(datetime.now(),i-num_steps_burn_in,duration))
                total_duration += duration
                total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print("%s：%s across %d steps,%.3f +/- %.3f sec / batch"%(datetime.now(),info_string,
                                                             num_batches,mn,sd))
batch_size = 32
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],dtype=tf.float32,stddev=0.1))
        keep_prob = tf.placeholder(tf.float32)
        predictions,softmax,fc8,p=inference_op(images,keep_prob)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        time_tensorflow_run(sess,predictions,{keep_prob:1.0},"Forward")
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective,p)
        time_tensorflow_run(sess,grad,{keep_prob:0.5},"Forward-backward")

if __name__ == "__main__":
    run_benchmark()


