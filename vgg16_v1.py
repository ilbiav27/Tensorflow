import inspect
import os
import matplotlib.pyplot as plt


import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]
printNUM = 0
NUM_CLASSES = 21
batch_size = 4
std = 1/192.0

class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        self.printNUM = printNUM
        self.cost = 0.0
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            #path of vgg16_weight
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")
    '''        
    #print filter using imshow
    def filter_print(self, fig, i, img, name):
        ax = fig.add_subplot(1, 5, 1 + i, xticks=[], yticks=[])
        ax.set_title(name)
        plt.imshow(img)
        
    #print filter using tensorboard
    def tensorboard_print(self, dim, value, name):
        img1,img2 = tf.split(axis=3, num_or_size_splits=[1,dim-1], value=value)
        tf.summary.image(name,img1,3)
    '''
    def build(self, rgb, labels):

        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [256, 256, 1]
        assert green.get_shape().as_list()[1:] == [256, 256, 1]
        assert blue.get_shape().as_list()[1:] == [256, 256, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ], name = 'input')
        assert bgr.get_shape().as_list()[1:] == [256, 256, 3]
        
        labels = tf.reshape(tf.one_hot(labels, NUM_CLASSES,
           on_value=1.0, off_value=0.0,
           axis=-1),[-1,NUM_CLASSES])
        
        with tf.name_scope("conv1"):
            # 256 * 256 * batch_size
            self.conv1_1 = self.conv_layer(bgr, "conv1_1")
            self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        with tf.name_scope("conv2"):
            # 128 * 128 * 64
            self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
            self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        with tf.name_scope("conv3"):
            # 64 * 64 * 128
            self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
            self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
            self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
            self.pool3 = self.max_pool(self.conv3_3, 'pool3')
            
        with tf.name_scope("conv4"):
            # 32 * 32 * 256
            self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
            self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
            self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
            self.pool4 = self.max_pool(self.conv4_3, 'pool4')
            
        with tf.name_scope("conv5"):
            # 16 * 16 * 512
            self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
            self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
            self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
            self.pool5 = self.max_pool(self.conv5_3, 'pool5')
            
        # 8 * 8 * 512 = 32768
        shape = self.pool5.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        self.temp = tf.reshape(self.pool5, [-1, dim])
            
        with tf.name_scope("fc6"):
            weight = self._variable_with_weight_decay(name = 'fc6_weight', shape = [dim, 4096],stddev = std, wd = None)
            biases = tf.Variable(tf.random_normal([4096]),name = 'fc6_biases')
            self.fc6 = tf.nn.bias_add(tf.matmul(self.temp, weight), biases)
            self.relu6 = tf.nn.relu(self.fc6)
            
        with tf.name_scope("fc7"):
            weight_ = self._variable_with_weight_decay(name = 'fc7_weight', shape = [4096, 1000],stddev=std, wd=None)
            biases_ = tf.Variable(tf.random_normal([1000]),name = 'fc7_biases')
            self.fc7 = tf.nn.bias_add(tf.matmul(self.relu6, weight_), biases_)
            self.relu7 = tf.nn.relu(self.fc7)
            
        with tf.name_scope("fc8"):
            weight__ = self._variable_with_weight_decay(name = 'fc8_weight', shape = [1000, NUM_CLASSES],stddev=std, wd=None)
            biases__ = tf.Variable(tf.random_normal([NUM_CLASSES]), name = 'fc8_biases')
            self.fc8 = tf.nn.bias_add(tf.matmul(self.relu7, weight__), biases__, name = 'fc8_matmul')
            self.prob = tf.nn.softmax(self.fc8, name="probability")
            
        with tf.name_scope('cost'):
            self.cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.fc8, labels = labels, name = 'cross_entropy_cost') 
            self.train = tf.train.AdamOptimizer(1e-4).minimize(self.cost)
            
        # 정확도를 계산하는 연산을 추가합니다.
        with tf.name_scope('accuracy'):
            self.correct_prediction = tf.equal(tf.argmax(self.prob, 1), tf.argmax(labels,1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.data_dict = None
            print(("build model finished: %ds" % (time.time() - start_time)))
    
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        #self.filter_print(self.fig, self.printNUM, bottom[0], name)
        #self.printNUM += 1
        
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            
            #The shape of filter -ex)3,224,224,64
            shape = relu.get_shape().as_list()
            #img1,img2 = tf.split(axis=3, num_or_size_splits=[1,relu.get_shape().as_list()[3]-1], value=relu)
            
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")
    
    def _variable_on_cpu(self, name, shape, initializer):
        with tf.device('/cpu:0'):
            dtype = tf.float32
            var = tf.get_variable(name, shape, initializer=initializer,  dtype=dtype)
        return var


    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        dtype = tf.float32
        var = self._variable_on_cpu(name,shape,initializer=tf.contrib.layers.xavier_initializer())
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var
