import inspect
import os

import numpy as np
import tensorflow as tf
import time
import var

# VGG_MEAN = [103.939, 116.779, 123.68] # BGR


VGG_MEAN = [116.779, 123.68, 103.939] # RGB from chris


class Vgg16:
    def __init__(self, X1, isTrain):




        self.data_dict =self.init_weightsss(isTrain)
        self.isTrain = isTrain
        self.X = X1
        self.weight_dict= {}


        if isTrain:
            self.fc9_W = self.init_weights([1000,2])
            self.fc9_b = self.init_weights([2])

        # self.init_weightsss(isTrain)

        print("npy file loaded")


    def init_weightsss(self, isTrain):

        if isTrain:
            data_dict = np.load(var.WEIGHT_PATH)
        else:
            data_dict = np.load(var.TUNED_WEIGHT_PATH)

        keys = sorted(data_dict.keys())
        for i, k in enumerate(keys):
            print(i, k, data_dict[k].shape)
        return data_dict



    def init_weights(self, shape, given_dev=0.01):
        return tf.Variable(tf.random_normal(shape, stddev=given_dev))

    def get_weight_path(self, isTrain):
        pass

    def build(self):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        # bgr_ = bgr*255.0
        bgr_= self.X
        start_time = time.time()
        print("build model started")

        # blue ,green, red = tf.split(axis=3, num_or_size_splits=3, value= bgr)
        red  ,green, blue, = tf.split(axis=3, num_or_size_splits=3, value= bgr_)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            # blue - VGG_MEAN[0],
            # green - VGG_MEAN[1],
            # red - VGG_MEAN[2],

            red - VGG_MEAN[0],
            green - VGG_MEAN[1],
            blue - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]



        print(bgr.shape)

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')




        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')





        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        # self.fc9 = self.fc_layer(self.fc8,'fc9')
        # self.relu9 = tf.nn.relu(self.fc9)




        relu8 = tf.nn.relu(self.fc8)
        fc9 = self.fc_layer(relu8, 'fc9')
        print(("build model finished: %ds" % (time.time() - start_time)))
        return fc9

        # self.prob = tf.nn.softmax(self.fc8, name="prob")




    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_abc(var.SIG_WEIGHT, name, var.FILTER)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_abc(var.SIG_BIAS, name,var.BIAS)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_abc(var.SIG_WEIGHT, name, var.WEIGHT)
            biases = self.get_abc(var.SIG_BIAS, name, var.BIAS)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_variable(self, shape, name):
        return tf.Variable(tf.truncated_normal(shape, dtype=tf.float32,
                                                 stddev=1e-1), name=name)


    def get_abc(self, sig, layer_name, var_name):
        key = layer_name+ sig


        if self.isTrain:
            if layer_name=='fc9':
                if var_name == var.WEIGHT:
                    w = self.fc9_W
                else:
                    w = self.fc9_b

            else:
                _weight = self.data_dict[key]
                shape = list(_weight.shape)
                w = self.get_variable(shape, var_name)
            self.weight_dict[key]= w

        else:
            _weight = self.data_dict[key]
            w = tf.constant(_weight, name="filter")

        return w
    #
    # def get_bias(self, name):
    #     name_ = name + '_b'
    #     if self.isTrain and name=='fc9':
    #         return self.fc9_b
    #     else:
    #         return tf.constant(self.data_dict[name_], name="biases")
    #
    # def get_fc_weight(self, name):
    #     name_ = name + '_W'
    #     if self.isTrain:
    #         if name=='fc9':
    #             return self.fc9_W
    #         else:
    #             _weight = self.data_dict[name_]
    #             shape = _weight.shape.aslist()
    #
    #     else:
    #         return tf.constant(self.data_dict[name_], name="weights")

    def assigne(self, sess):

        keys = sorted(self.data_dict.keys())
        for i, k in enumerate(keys):
            # print (i, k, np.shape(weights[k]))
            # print(k, '=weights[', i, ']')
            sess.run(self.weight_dict[k].assign(self.data_dict[k]))
        print('all weights loaded')

