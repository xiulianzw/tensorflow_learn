import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
#用来正常显示中文
plt.rcParams["font.sans-serif"]=["SimHei"]

def img_random_crop():
    img = cv2.imread("img/img.jpg")
    #将图片进行随机裁剪为280×280
    crop_img = tf.random_crop(img,[280,280,3])
    sess = tf.InteractiveSession()
    #显示图片
    # cv2.imwrite("img/crop.jpg",crop_img.eval())
    plt.figure(1)
    plt.subplot(121)
    #将图片由BGR转成RGB
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("原始图片")
    plt.subplot(122)
    crop_img = cv2.cvtColor(crop_img.eval(),cv2.COLOR_BGR2RGB)
    plt.title("裁剪后的图片")
    plt.imshow(crop_img)
    plt.show()
    sess.close()

def random_flip():
    img = cv2.imread("img/img.jpg")
    #将图片随机进行水平翻转
    h_flip_img = tf.image.random_flip_left_right(img)
    #将图片随机进行垂直翻转
    v_flip_img = tf.image.random_flip_up_down(img)
    sess = tf.InteractiveSession()
    #通道转换
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h_flip_img = cv2.cvtColor(h_flip_img.eval(),cv2.COLOR_BGR2RGB)
    v_flip_img = cv2.cvtColor(v_flip_img.eval(),cv2.COLOR_BGR2RGB)
    #显示图片
    plt.figure(1)
    plt.subplot(131)
    plt.title("水平翻转")
    plt.imshow(h_flip_img)
    plt.subplot(132)
    plt.title("垂直翻转")
    plt.imshow(v_flip_img)
    plt.subplot(133)
    plt.title("原始图片")
    plt.imshow(img)
    plt.show()

def random_hug_stat():
    img = cv2.imread("img/img.jpg")
    #随机设置图片的亮度
    random_brightness = tf.image.random_brightness(img,max_delta=30)
    #随机设置图片的对比度
    random_contrast = tf.image.random_contrast(img,lower=0.2,upper=1.8)
    #随机设置图片的色度
    random_hue = tf.image.random_hue(img,max_delta=0.3)
    #随机设置图片的饱和度
    random_satu = tf.image.random_saturation(img,lower=0.2,upper=1.8)
    sess = tf.InteractiveSession()
    #转换通道
    random_brightness = cv2.cvtColor(random_brightness.eval(),cv2.COLOR_BGR2RGB)
    random_contrast = cv2.cvtColor(random_contrast.eval(),cv2.COLOR_BGR2RGB)
    random_hue = cv2.cvtColor(random_hue.eval(),cv2.COLOR_BGR2RGB)
    random_satu = cv2.cvtColor(random_satu.eval(),cv2.COLOR_BGR2RGB)
    #显示图片
    plt.figure(1)
    plt.subplot(221)
    plt.title("随机亮度")
    plt.imshow(random_brightness)
    plt.subplot(222)
    plt.title("随机对比度")
    plt.imshow(random_contrast)
    plt.subplot(223)
    plt.title("随机色度")
    plt.imshow(random_hue)
    plt.subplot(224)
    plt.title("随机饱和度")
    plt.imshow(random_satu)
    plt.show()


if __name__ == "__main__":
    img = cv2.imread("img/img.jpg")
    #将图片进行标准化
    std_img = tf.image.per_image_standardization(img)
    sess = tf.InteractiveSession()
    print(std_img.eval())
