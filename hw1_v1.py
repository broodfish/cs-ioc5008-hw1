import cv2
from cv2 import cv2 as cv 
import PIL.Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import math

spatial_pool_size = [4, 2, 1]
spatial_pool_dim = sum([i*i for i in spatial_pool_size])
n_classes = 13

# 讀檔
def get_training_file(data_dir):
    images = []
    subfolders = []
    subfolder_names = []
    
    for dirPath, dirNames, fileNames in os.walk(data_dir):
        for name in fileNames:
            images.append(os.path.join(dirPath, name))
        
        for name in dirNames:
            subfolder_names.append(name)
            subfolders.append(os.path.join(dirPath, name))

            
    labels = []
    count = 0
    for a_folder in subfolders:
        n_img = len(os.listdir(a_folder))
        labels = np.append(labels, n_img * [count])
        count+=1
    
    subfolders = np.array([images, labels])
    subfolders = subfolders.transpose()
    
    image_list = list(subfolders[:, 0])
    label_list = list(subfolders[:, 1])
    label_list = [int(float(i)) for i in label_list]
    print("\nReading training data finished.")
    return image_list, label_list

def get_testing_file(data_dir):
    images = []
    
    for dirPath, dirNames, fileNames in os.walk(data_dir):
        for name in fileNames:
            images.append(os.path.join(dirPath, name))

    image_list = list(images)
    print("\nReading testing data finished.")
    return image_list

def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list = tf.train.Int64List(value = value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

# 轉成TFRecord
def convert_to_TFRecord(images, labels, filename):
    n_samples = len(labels)
    TFWriter = tf.python_io.TFRecordWriter(filename+'.tfrecords')
    
    print('\nTransform start...')
    for i in np.arange(0, n_samples):
        try:
            image = cv.imread(images[i])
            height = image.shape[0]
            width = image.shape[1]
            if image is None:
                print('Error image:' + images[i])
            else:
                image_raw = image.tostring()
            label = labels[i]
            
            ftrs = tf.train.Features(
                    feature = {'Label': int64_feature(label),
                               'image_raw': bytes_feature(image_raw),
                               'height': int64_feature(height),
                               'width': int64_feature(width)})
            example = tf.train.Example(features=ftrs)
            TFWriter.write(example.SerializeToString())
        except IOError as e:
            print('Skip!\n')
    TFWriter.close()
    print('Transform done!')

def convert_to_TFRecord2(images, filename):
    n_samples = len(images)
    TFWriter = tf.python_io.TFRecordWriter(filename+'.tfrecords')
    
    print('\nTransform start...')
    for i in np.arange(0, n_samples):
        try:
            image = cv.imread(images[i])
            height = image.shape[0]
            width = image.shape[1]
            if image is None:
                print('Error image:' + images[i])
            else:
                image_raw = image.tostring()
            
            ftrs = tf.train.Features(
                    feature = {'image_raw': bytes_feature(image_raw),
                               'height': int64_feature(height),
                               'width': int64_feature(width)})
            example = tf.train.Example(features=ftrs)
            TFWriter.write(example.SerializeToString())
        except IOError as e:
            print('Skip!\n')
    TFWriter.close()
    print('Transform done!')

# 讀&解碼TFrecord
def read_and_decode(filename):
    print('\nDecode Start...')
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                          features={
                                  'Label':tf.FixedLenFeature([], tf.int64),
                                  'image_raw':tf.FixedLenFeature([], tf.string),
                                  'height': tf.FixedLenFeature([], tf.int64),
                                  'width': tf.FixedLenFeature([], tf.int64)
                                  })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['Label'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    image = tf.reshape(image, [height, width, 3])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    print("Decode DONE!")
    return image, label, height, width

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)#隨機變量
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1  !! the note from Document
	
    # x 是指圖片數值
    # W 是指weight
    # strides 是指步長，需輸入要是四個維度，而第一個維度與最後一個維度必須為1。第二維度是指X方向，第三維度則是Y方向
    # padding方式，padding='SAME' or 'VALID'
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_4x4(x):
    # stride [1, x_movement, y_movement, 1]
    # 用來避免strides過大導致丟失太多特徵
    # 這邊可以選用 tf.nn.avg_pool 或是 tf.nn.max_pool，官方範例是用 tf.nn.max_pool
    # strides 是指步長，需輸入要是四個維度，而第一個維度與最後一個維度必須為1。第二維度是指X方向，第三維度則是Y方向
    # strides 第二維度與第三維度設定4是為了減小圖像大小
    return tf.nn.max_pool(x, ksize=[1,4,4,1], strides=[1,4,4,1], padding='VALID')

def Spp_layer(feature_map, spatial_pool_size):
    print(feature_map.get_shape())
    ############### get feature size ##############
    height=int(feature_map.get_shape()[1])
    width=int(feature_map.get_shape()[2])
    ############### get batch size ##############
    batch_num = 50

    for i in range(len(spatial_pool_size)):
        # calculate each pooling layer
        ############### stride ############## 
        stride_h = int(np.ceil(height/spatial_pool_size[i]))
        stride_w = int(np.ceil(width/spatial_pool_size[i]))
        ############### kernel ##############
        window_w = int(np.ceil(width/spatial_pool_size[i]))
        window_h = int(np.ceil(height/spatial_pool_size[i]))
        #print(feature_map.get_shape())
        ############### max pool ##############
        pooling_out = tf.nn.max_pool(feature_map, ksize=[1, window_h, window_w, 1], strides=[1, stride_h, stride_w, 1],padding='SAME')
        #print(pooling_out.get_shape())
        if i == 0:
            spp = tf.reshape(pooling_out, [batch_num, -1])
        else:
            ############### concat each pool result ##############
            spp = tf.concat(axis=1, values=[spp, tf.reshape(pooling_out, [batch_num, -1])])

    return spp

##=====train======##
def train(data_dir):
    # train your model with images from data_dir
    # the following code is just a placeholder
    # 讀data
    #image_list, label_list = get_training_file(data_dir)
    #convert_to_TFRecord(image_list, label_list, data_dir)
    image, label, height, width = read_and_decode(data_dir + ".tfrecords")
    image = tf.image.resize_images(image, [256, 256])
    image_train, label_train = tf.train.shuffle_batch([image, label], batch_size = 50, capacity = 10000, num_threads = 1, min_after_dequeue = 1000)
    label_train = tf.one_hot(label_train, n_classes)
    
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 256, 256, 3], name = 'xs')
    ys = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

    height = int(xs.get_shape()[1])
    width = int(xs.get_shape()[2])
    x_image = tf.reshape(xs, [-1, height, width, 3])
    # -1是指放棄資料原有的所有維度，256,256則是新給維度，1則是指說資料只有一個數值(黑白)，若是彩色則為3(RGB)
    
    # CNN model
    
    ##conv1 layer##
    W_conv1 = weight_variable([11, 11, 3, 32]) # patch 11x11, in size 3(image的厚度), out size 96
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) ## conv
    h_pool1 = max_pool_4x4(h_conv1) ## pool
    
    ##conv2 layer##
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) ## conv
    h_pool2 = Spp_layer(h_conv2, spatial_pool_size)  ## Spp
    
    ##func1 layer##
    W_fc1 = weight_variable([spatial_pool_dim*64,512])
    b_fc1 = bias_variable([512])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, spatial_pool_dim*64]) # [n_samples, 16, 16, 128] -> [n_samples, 16*16*128]
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) ## dropout
    
    ##func2 layer##
    W_fc2 = weight_variable([512, n_classes])
    b_fc2 = bias_variable([n_classes])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1])) # loss
    
    tf.add_to_collection('pred_network', prediction)
    
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        if os.path.exists('./net/train_model.ckpt'):
            saver.restore(sess, './net/train_model.ckpt')
        else:
            # 初始化
            init=tf.global_variables_initializer()
            sess.run(init)
        
        # 建立執行緒協調器
        coord = tf.train.Coordinator()
        # 啟動文件對列，開始讀取文件
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        try:
            print("Start Training!")
            print('************')
            for i in range(10000):
                image_train_batch, label_train_batch = sess.run([image_train, label_train])
                sess.run(train_step, feed_dict = {xs: image_train_batch, ys: label_train_batch, keep_prob: 0.5})
                if i % 10 == 0:
                    train_accuracy = accuracy.eval(feed_dict={xs: image_train_batch, ys: label_train_batch, keep_prob: 1.0})
                    print('Iter %d, accuracy %4.2f%%' % (i, train_accuracy*100))
            print("Finish Training!")
            
        except tf.errors.OutOfRangeError:
            print('DONE!')
        finally:
            coord.request_stop()
                
        coord.join(threads)
        
        saver_path = saver.save(sess, './net/train_model.ckpt')
        print("model saved in file: ", saver_path)
        
    pass

def test(data_dir):
    # make your model give prediction for images from data_dir
    # the following code is just a placeholder

    #image_list = get_testing_file(data_dir)
    #print(len(image_list))
    #convert_to_TFRecord2(image_list, data_dir)
    print('\nDecode Start...')
    filename_queue = tf.train.string_input_producer([data_dir + ".tfrecords"], num_epochs=None)
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                          features={
                                  'image_raw':tf.FixedLenFeature([], tf.string),
                                  'height': tf.FixedLenFeature([], tf.int64),
                                  'width': tf.FixedLenFeature([], tf.int64)
                                  })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    image = tf.reshape(image, [height, width, 3])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    print("Decode DONE!")
    image = tf.image.resize_images(image, [256, 256])
    print(image.get_shape())
    #input_queue = tf.train.slice_input_producer([image], shuffle = False, num_epochs = 1)
    image_test_batch = tf.train.batch([image], batch_size = 50, capacity = 1040, num_threads = 1, allow_smaller_final_batch=True)
    print(image_test_batch.get_shape())
  
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 256, 256, 3])   # 256x256
    keep_prob = tf.placeholder(tf.float32)

    print("Start Tseting!")
    print('************')
    
    p=[]
    fp = open("label.txt", "w")
    with tf.Session() as sess:
        # 初始化
        init=tf.local_variables_initializer()
        sess.run(init)

        saver = tf.train.import_meta_graph("./net/train_model.ckpt.meta")
        saver.restore(sess, "./net/train_model.ckpt")

        graph = tf.get_default_graph()
        xs = graph.get_operation_by_name('xs').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        prediction = tf.get_collection("pred_network")[0]

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for i in range(0,21):
                image_test = sess.run([image_test_batch])
                pred = sess.run(prediction, feed_dict={xs: image_test[0], keep_prob: 1.0})
                p.append(np.argmax(pred, axis = 1))
                p_list = (np.argmax(pred, axis = 1)).tolist()
                if i % 10 == 0:
                    print('Iter %d' % (i))
                print(p_list)
                fp.writelines(["%s\n" % item  for item in p_list])
            print("Finish Prediction!")
        except tf.errors.OutOfRangeError:
                print('DONE!')
        finally:
            print("STOP!")
            coord.request_stop()
        #coord.join(threads)
        print(p)
        fp.close()

    pass

