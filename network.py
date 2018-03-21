import tensorflow as tf
import numpy as np

class Network():
    def __init__(self,img_width,img_height,name="network"):
        self.name = name
        self.img_width = img_width
        self.img_height = img_width

    def net(self):
        with(tf.variable_scope(self.name)) as scope :
            self.input_state = tf.placeholder(tf.float32,[None,self.img_width,self.img_height,4],name="input_state")

            input_layer = tf.nn.conv2d(self.input_state,32,8,8,4,4,padding="SAME")
            input_max_pooled = tf.nn.max_pool(input_layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
            layer2 = tf.nn.conv2d(input_max_pooled,64,4,4,4,2,2,padding="SAME")
            layer3 = tf.nn.conv2d(layer2,64,3,3,1,1,padding="SAME")

            shape=layer3.get_shape().as_list()
            layer3= tf.reshape(layer3, [-1, shape[1] * shape[2] * shape[3]])
            shape = layer3.get_shape().as_list()

            o_l1 = tf.nn.relu(linear1d(layer3, shape[1], 512))

            self.q_values = linear1d(o_l1, 512, 2)




