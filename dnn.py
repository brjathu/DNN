# Copyright (c) 2017
#
# Jathushan.Rajasegaran@gmail.com


import os
import tensorflow as tf
from time import localtime, strftime
import numpy as np
import matplotlib.pyplot as plt


tf.set_random_seed(1)
tf.logging.set_verbosity(tf.logging.ERROR)
print(tf.__version__)


# parameters
LR = 0.001
EPOCH = 500
BATCH_SIZE = 500
MODEL = 5
RESULTS_PATH = "./model/" + str(MODEL) + "/"
LOG_FILE = open(RESULTS_PATH + 'log.txt', 'a')


def logEntry(TMP_STRING):
    LOG_FILE.write(TMP_STRING)
    LOG_FILE.write("\n")
    LOG_FILE.flush()
    print(TMP_STRING)


def buildTestSet(fileName):
    logEntry(strftime("%Y-%m-%d %H:%M:%S", localtime()) +
             " **** Building Dataset")
    X = []
    Y = []
    with open(fileName) as file:
        for i in file:
            c = i.rstrip()
            y = c[-30:]
            x = c[:-30]

            x_list = eval(x)
            y = eval(y)

            X.append(x_list)
            Y.append(y)

    X = np.asarray(X)
    logEntry(strftime("%Y-%m-%d %H:%M:%S", localtime()) +
             " **** Done Building Dataset")
    return X, Y


tf.reset_default_graph()

# tf placeholder
# value in the range of (0, 1)
tf_x = tf.placeholder(tf.float32, [None, 96 * 30])
tf_y = tf.placeholder(tf.float32, [None, 10])

# model 1. 2
# hidden_layer1 = tf.layers.dense(tf_x, 2000, tf.nn.relu)
# hidden_layer2 = tf.layers.dense(hidden_layer1, 1000, tf.nn.relu)
# hidden_layer3 = tf.layers.dense(hidden_layer2, 500, tf.nn.relu)
# hidden_layer4 = tf.layers.dense(hidden_layer3, 250, tf.nn.relu)
# hidden_layer5 = tf.layers.dense(hidden_layer4, 125, tf.nn.relu)
# hidden_layer6 = tf.layers.dense(hidden_layer5, 50, tf.nn.relu)
# hidden_layer7 = tf.layers.dense(hidden_layer6, 10, tf.nn.relu)


# model 3
# hidden_layer1 = tf.layers.dense(tf_x, 5000, tf.nn.relu)
# hidden_layer2 = tf.layers.dense(hidden_layer1, 2000, tf.nn.relu)
# hidden_layer3 = tf.layers.dense(hidden_layer2, 1000, tf.nn.relu)
# hidden_layer4 = tf.layers.dense(hidden_layer3, 2000, tf.nn.relu)
# hidden_layer5 = tf.layers.dense(hidden_layer4, 1000, tf.nn.relu)
# hidden_layer6 = tf.layers.dense(hidden_layer5, 500, tf.nn.relu)
# hidden_layer7 = tf.layers.dense(hidden_layer6, 100, tf.nn.relu)
# hidden_layer8 = tf.layers.dense(hidden_layer7, 10, tf.nn.relu)


# model 4
hidden_layer1 = tf.layers.dense(tf_x, 2500, tf.nn.tanh, name="hidden_layer1")
hidden_layer2 = tf.layers.dense(hidden_layer1, 2000, tf.nn.tanh, name="hidden_layer2")
hidden_layer3 = tf.layers.dense(hidden_layer2, 1500, tf.nn.tanh, name="hidden_layer3")
# dropout3 = tf.layers.dropout(hidden_layer3, rate=0.5, name= "dropout3")
hidden_layer4 = tf.layers.dense(hidden_layer3, 1000, tf.nn.tanh, name="hidden_layer4")
# dropout4 = tf.layers.dropout(hidden_layer4, rate=0.5, name= "dropout4")
hidden_layer5 = tf.layers.dense(hidden_layer4, 500, tf.nn.tanh, name="hidden_layer5")
hidden_layer6 = tf.layers.dense(hidden_layer5, 200, tf.nn.tanh, name="hidden_layer6")
hidden_layer7 = tf.layers.dense(hidden_layer6, 50, tf.nn.tanh, name="hidden_layer7")
hidden_layer8 = tf.layers.dense(hidden_layer7, 10, tf.nn.tanh, name="hidden_layer8")

softmax_layer = tf.nn.softmax(hidden_layer8, name="softmax")

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=softmax_layer, labels=tf_y))
# loss = tf.losses.mean_squared_error(labels=tf_y, predictions=softmax_layer)
train = tf.train.AdamOptimizer(LR).minimize(loss)


# prepare data
# x_train, y_train = buildTestSet("./data/data_train_balanced.txt")
# x_train = np.array(x_train)
# y_train = np.array(y_train)

# #preprocessing
# X = []
# Y =[]
# for i in range(x_train.shape[0]):
# 	X.append( np.reshape(x_train[i],(1,-1))[0])
# 	Y.append( np.reshape(y_train[i],(1,-1))[0])

# np.save("./data/x_train.npy", np.array(X))
# np.save("./data/y_train.npy", np.array(Y))

# load data
x = np.load("./data/x_train.npy")
y = np.load("./data/y_train.npy")

ratio = int(x.shape[0] * 80 / 100)
x_train = x[:ratio, :]
y_train = y[:ratio, :]
print(x_train.shape)
print(y_train.shape)

loss_graph = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch in range(EPOCH):
        loss_avg = 0
        for i in range(int(x_train.shape[0] / BATCH_SIZE)):
            _, loss_val = sess.run([train, loss], {tf_x: np.reshape(x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE], (
                BATCH_SIZE, 2880)), tf_y: np.reshape(y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE], (BATCH_SIZE, 10))})
            loss_avg += loss_val
        print(loss_avg / i)
        loss_graph.append(loss_avg / i)
        np.save("./model/" + str(MODEL) + "/train.npy", np.array(loss_graph))
        plt.plot(np.array(loss_graph))
        plt.savefig("./model/" + str(MODEL) + "/loss_graph.png")
        # plt.show()

        saver.save(sess, "model/" + str(MODEL) + "/ckpt/dnn.ckpt")
