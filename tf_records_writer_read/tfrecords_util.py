import math
import os
import sys
import tensorflow as tf
from inception_preprocessing import preprocess_image

slim = tf.contrib.slim

#用来保存标签名称与数字标签的对应关系
LABELS_FILENAME = 'labels.txt'

def int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

#将图片信息转换为tfrecords可以保存的序列化信息
def image_to_tfexample(split_name,image_data, image_format, height, width, img_info):
    '''
    :param split_name: train或val或test
    :param image_data: 图片的二进制数据
    :param image_format: 图片的格式
    :param height: 图片的高
    :param width: 图片的宽
    :param img_info: 图片的标签或图片的名称,当split_name为test时,img_info为图片的名称否则为图片标签
    :return:
    '''
    if split_name == "test":
        return tf.train.Example(features=tf.train.Features(feature={
              'image/encoded': bytes_feature(image_data),
              'image/format': bytes_feature(image_format),
              'image/img_name': bytes_feature(img_info),
              'image/height': int64_feature(height),
              'image/width': int64_feature(width),
          }))
    else:
          return tf.train.Example(features=tf.train.Features(feature={
              'image/encoded': bytes_feature(image_data),
              'image/format': bytes_feature(image_format),
              'image/label': int64_feature(img_info),
              'image/height': int64_feature(height),
              'image/width': int64_feature(width),
          }))


def write_label_file(labels_to_class_names, dataset_dir,filename=LABELS_FILENAME):
    '''将标签与类标名称的对应关系保存为文件
    :param labels_to_class_names: 标签与类标名称的对饮
    :param dataset_dir: 文件的保存目录
    :param filename: 文件保存的名称
    :return: 空
    '''
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
    '''判断标签与类标名称对应关系的文件是否存在
    :param dataset_dir: 文件的保存目录
    :param filename: 文件名称
    :return: 空
    '''
    return tf.gfile.Exists(os.path.join(dataset_dir, filename))

#图片的读取类用来读取图片
class ImageReader(object):

  #将string类型的图片数据转换为jpg的3通道图片
  def __init__(self):
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  #获取图片的宽和高
  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  #将图片解码成为jpg格式
  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

'''
获取训练集的图片路径和图片的标签
'''
def _get_dateset_imgPaths(dataset_dir,split_name):
  '''
  获取图片的路径和图片对应的标签
  :param dataset_dir:数据存放的目录
  :return:猫的路径列表,狗的路径列表
  '''
  #保存所有的图片路径
  cat_img_paths = []
  dog_img_paths = []
  #获取文件所在路径
  dataset_dir = os.path.join(dataset_dir,split_name)
  #遍历目录下的所有图片
  for filename in os.listdir(dataset_dir):
      #获取文件的路径
      file_path = os.path.join(dataset_dir,filename)
      if file_path.endswith("jpg") and os.path.exists(file_path):
          #获取类别的名称
          label_name = filename.split(".")[0]
          if label_name == "cat":
              cat_img_paths.append(file_path)
          elif label_name == "dog":
              dog_img_paths.append(file_path)
  return cat_img_paths,dog_img_paths

#获取tfrecords文件的保存路径
def _get_dataset_filename(dataset_dir, split_name, shard_id, tfrecord_filename, _NUM_SHARDS):
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
      tfrecord_filename, split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_tfrecord_dataset(split_name, filenames, label_name_to_id, dataset_dir, tfrecord_filename, _NUM_SHARDS):
    '''
    :param split_name:train或val或test
    :param filenames:图片的路径列表
    :param label_name_to_id:标签名与数字标签的对应关系
    :param dataset_dir:数据存放的目录
    :param tfrecord_filename:文件保存的前缀名
    :param _NUM_SHARDS:将整个数据集分为几个文件
    :return:
    '''
    assert split_name in ['train', 'val','test']
    #计算平均每一个tfrecords文件保存多少张图片
    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))
    with tf.Graph().as_default():
        image_reader = ImageReader()
        with tf.Session('') as sess:
            for shard_id in range(_NUM_SHARDS):
                #获取tfrecord文件的名称
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id, tfrecord_filename = tfrecord_filename, _NUM_SHARDS = _NUM_SHARDS)
                #写tfrecords文件
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        #更新控制台中已经完成的图片数量
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i+1, len(filenames), shard_id))
                        sys.stdout.flush()
                        #读取图片,将图片数据读取为bytes
                        image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                        #获取图片的高和宽
                        height, width = image_reader.read_image_dims(sess, image_data)
                        #获取路径中的图片名称
                        img_name = os.path.basename(filenames[i])
                        if split_name == "test":
                            #需要将图片名称转换为二进制
                            example = image_to_tfexample(
                                split_name,image_data, b'jpg', height, width, img_name.encode())
                            tfrecord_writer.write(example.SerializeToString())
                        else:
                            #获取图片的类别
                            class_name = img_name.split(".")[0]
                            label_id = label_name_to_id[class_name]
                            example = image_to_tfexample(
                                split_name,image_data, b'jpg', height, width, label_id)
                            tfrecord_writer.write(example.SerializeToString())
                sys.stdout.write('\n')
                sys.stdout.flush()

#判断tfrecord文件是否已经存在
def _dataset_exists(dataset_dir, _NUM_SHARDS, output_filename):
    for split_name in ['train', 'val','test']:
        for shard_id in range(_NUM_SHARDS):
            tfrecord_filename = _get_dataset_filename(
                dataset_dir, split_name, shard_id, output_filename, _NUM_SHARDS)
            if not tf.gfile.Exists(tfrecord_filename):
                return False
    return True

#利用tfrecord文件来创建数据集
def get_dataset_by_tfrecords(split_name,dataset_dir,tfrecord_filename,num_classes,labels_to_name=None):
    if split_name not in ['train', 'val',"test"]:
        raise ValueError("the split_name value %s is error."%split_name)
    file_path_pattern = tfrecord_filename + "_" + split_name
    #获取满足条件的tfrecord文件
    tfrecord_paths = [os.path.join(dataset_dir,filename) for filename in os.listdir(dataset_dir)
                      if filename.startswith(file_path_pattern)]
    #统计图片的数量
    num_imgs = 0
    for tfrecord_file in tfrecord_paths:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_imgs += 1

    #创建一个tfrecord读文件对象
    reader = tf.TFRecordReader

    #定义需要读取图片中的数据
    if split_name == "test":
        keys_to_feature = {
            "image/encoded":tf.FixedLenFeature((),tf.string,default_value=""),
            "image/format":tf.FixedLenFeature((),tf.string,default_value="jpg"),
            "image/img_name":tf.FixedLenFeature((),tf.string,default_value="")
        }
        items_to_handles = {
            "image":slim.tfexample_decoder.Image(),
            "img_name":slim.tfexample_decoder.Tensor("image/img_name"),
        }
        items_to_descriptions = {
            "image":"a 3-channel RGB image",
            "img_name":"a image name"
        }
         #创建一个tfrecoder解析对象
        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_feature,items_to_handles)
        #读取所有的tfrecord文件，创建数据集
        dataset = slim.dataset.Dataset(
            data_sources = tfrecord_paths,
            decoder = decoder,
            reader = reader,
            num_readers = 4,
            num_samples = num_imgs,
            num_classes = num_classes,
            items_to_descriptions=items_to_descriptions
        )
    else:
        keys_to_feature = {
            "image/encoded":tf.FixedLenFeature((),tf.string,default_value=""),
            "image/format":tf.FixedLenFeature((),tf.string,default_value="jpg"),
            "image/label":tf.FixedLenFeature([],tf.int64,default_value=tf.zeros([],tf.int64))
        }
        items_to_handles = {
            "image":slim.tfexample_decoder.Image(),
            "label":slim.tfexample_decoder.Tensor("image/label")
        }
        items_to_descriptions = {
            "image":"a 3-channel RGB image",
            "img_name":"a image label"
        }
        #创建一个tfrecoder解析对象
        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_feature,items_to_handles)
        #读取所有的tfrecord文件,创建数据集
        dataset = slim.dataset.Dataset(
            data_sources = tfrecord_paths,
            decoder = decoder,
            reader = reader,
            num_readers = 4,
            num_samples = num_imgs,
            num_classes = num_classes,
            labels_to_name = labels_to_name,
            items_to_descriptions = items_to_descriptions
        )
    return dataset

def load_batch(split_name,dataset,batch_size,height,width):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity = 24 + 3 * batch_size,
        common_queue_min = 24
    )
    if split_name == "test":
        raw_image,img_name = data_provider.get(["image","img_name"])
        image = preprocess_image(raw_image, height, width,False)
        #获取一个batch的数据
        images,img_names = tf.train.batch(
            [image,img_name],
            batch_size=batch_size,
            num_threads=4,
            capacity=4*batch_size,
            allow_smaller_final_batch=True
        )
        return images,img_names
    else:
        raw_image,img_label = data_provider.get(["image","label"])
        #Perform the correct preprocessing for this image depending if it is training or evaluating
        image = preprocess_image(raw_image, height, width,True)
        #As for the raw images, we just do a simple reshape to batch it up
        raw_image = tf.expand_dims(raw_image, 0)
        raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
        raw_image = tf.squeeze(raw_image)
        #获取一个batch数据
        images,raw_image,labels = tf.train.batch(
            [image,raw_image,img_label],
            batch_size=batch_size,
            num_threads=4,
            capacity=4*batch_size,
            allow_smaller_final_batch=True
        )
        return images,raw_image,labels
