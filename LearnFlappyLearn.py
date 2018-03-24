import tensorflow as tf
import cv2
import sys
sys.path.append("./game")
import wrapped_flappy_bird as game

Actions = 2

def createNeuralNet():

    #Initilizing weight values and biases of every layer

    weights_conv1 = tf.Variable(tf.random_normal([8,8,4,32], stddev=0.01),
                          name="weights")
    biases_conv1 = tf.Variable(tf.zeros([32]))

    weights_conv2 = tf.Variable(tf.random_normal([4,4,32,64], stddev=0.01),
                          name="weights")
    biases_conv2 = tf.Variable(tf.zeros([64]))

    weights_conv3 = tf.Variable(tf.random_normal([3,3,64,64], stddev=0.01),
                          name="weights")
    biases_conv3 = tf.Variable(tf.zeros([64]))

    weights_fc1 = tf.Variable(tf.random_normal([1600,512], stddev=0.01),
                          name="weights")
    biases_fc1 = tf.Variable(tf.zeros([512]))

    weights_fc2= tf.Variable(tf.random_normal([512,Actions], stddev=0.01),
                          name="weights")
    biases_fc2 = tf.Variable(tf.zeros([Actions]))

    #Network Input
    Input = tf.placeholder("float",[None,80,80,4])

    #Network layer
    layer1 = tf.nn.conv2d(input=input, filter=weights_conv1, strides=[1, 1, 1, 1], padding='SAME')
    layer1 += biases_conv1
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.nn.conv2d(input=layer1, filter=weights_conv2, strides=[1, 1, 1, 1], padding='SAME')
    layer2 += biases_conv2
    layer2 = tf.nn.relu(layer2)

    layer3 = tf.nn.conv2d(input=layer2, filter=weights_conv1, strides=[1, 1, 1, 1], padding='SAME')
    layer3 += biases_conv3
    layer3 = tf.nn.relu(layer3)

    layer3_flat = tf.reshape(layer3, [-1, 1600])

    layer_fc1 = tf.nn.relu(tf.matmul(layer3_flat, weights_fc1) + biases_fc1)

    # readout layer
    readout = tf.matmul(layer_fc1,weights_fc2) + biases_fc2

    return Input,readout,layer_fc1









