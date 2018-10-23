import tensorflow as tf
import numpy as np
from tfrecords_util import _get_dateset_imgPaths,_convert_tfrecord_dataset,get_dataset_by_tfrecords,load_batch
import matplotlib.pyplot as plt

#数据所在的目录路径
dataset_dir_path = "D:/dataset/kaggle/cat_or_dog/train"
#类标名称和数字的对应关系
label_name_to_num = {"cat":0,"dog":1}
label_num_to_name = {value:key for key,value in label_name_to_num.items()}
#设置验证集占整个数据集的比例
val_size = 0.2
batch_size = 1

#生成tfrecord文件
def generate_tfreocrd():
    #获取目录下所有的猫和狗图片的路径
    cat_img_paths,dog_img_paths = _get_dateset_imgPaths(dataset_dir_path,"train")
    #打乱路径列表的顺序
    np.random.shuffle(cat_img_paths)
    np.random.shuffle(dog_img_paths)
    #计算不同类别验证集所占的图片数量
    cat_val_num = int(len(cat_img_paths) * val_size)
    dog_val_num = int(len(dog_img_paths) * val_size)
    #将所有的图片路径分为训练集和验证集
    train_img_paths = cat_img_paths[cat_val_num:]
    val_img_paths = cat_img_paths[:cat_val_num]
    train_img_paths.extend(dog_img_paths[dog_val_num:])
    val_img_paths.extend(dog_img_paths[:dog_val_num])
    #打乱训练集和验证集的顺序
    np.random.shuffle(train_img_paths)
    np.random.shuffle(val_img_paths)
    #将训练集保存为tfrecord文件
    _convert_tfrecord_dataset("train",train_img_paths,label_name_to_num,dataset_dir_path,"catVSdog",2)
    #将验证集保存为tfrecord文件
    _convert_tfrecord_dataset("val",val_img_paths,label_name_to_num,dataset_dir_path,"catVSdog",1)

#读取tfrecord文件
def read_tfrecord():
    #从tfreocrd文件中读取数据
    train_dataset = get_dataset_by_tfrecords("train",dataset_dir_path,"catVSdog",2,label_num_to_name)
    images,raw_images,labels = load_batch("val",train_dataset,batch_size,227,227)
    with tf.Session() as sess:
        threads = tf.train.start_queue_runners(sess)
        for i in range(6):
            train_img,train_label = sess.run([raw_images,labels])
            plt.subplot(2,3,i+1)
            plt.imshow(np.array(train_img[0]))
            plt.title("image label:%s"%str(label_num_to_name[train_label[0]]))
        plt.show()

if __name__ == "__main__":
    # generate_tfreocrd()
    read_tfrecord()
