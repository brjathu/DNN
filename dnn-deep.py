# Copyright (c) 2017
#
# Jathushan.Rajasegaran@gmail.com


import os
import tensorflow as tf
from time import localtime, strftime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

tf.set_random_seed(1)
tf.logging.set_verbosity(tf.logging.ERROR)
print(tf.__version__)


# parameters
LR = 0.0005
EPOCH = 100
BATCH_SIZE = 1000


def logEntry(TMP_STRING):
    LOG_FILE.write(TMP_STRING)
    LOG_FILE.write("\n")
    LOG_FILE.flush()
    print(TMP_STRING)


def buildTestSet(fileName):
    print("start loading data")
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
    print("finished loading data")
    return X, Y


def plotConfusionMatrix(predictions, y, FILENAME, TITLE):
    LABELS = ['User 1', 'User 2', 'User 3', 'User 4', 'User 5', 'User 6', 'User 7', 'User 8', 'User 9', 'User 10']
    max_test = np.argmax(y, axis=1)
    max_predictions = np.argmax(predictions, axis=1)
    confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)
    accuracy = metrics.accuracy_score(max_test, max_predictions)
    logEntry("accuracy ==> " + str(accuracy))
    fig = plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
    plt.title("Confusion matrix: " + TITLE)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(FILENAME)
    plt.clf()
    plt.close('all')

    return accuracy


activation = ["tanh"]
layer_before_softmax = [64, 128]


# prepare data
x_train, y_train = buildTestSet("/flush1/sen040/96_tmp_data_augmented_final/deep/250/1.0/sc/train.txt")
# x_train, y_train = buildTestSet("/flush1/sen040/96_tmp_data_augmented_final/deep/250/1.0/sc/train.txt")
x_train = np.array(x_train)
y_train = np.array(y_train)

x_valid, y_valid = buildTestSet("/flush1/sen040/96_tmp_data_augmented_final/deep/250/1.0/sc/valid.txt")
# x_test, y_test = buildTestSet("/flush1/sen040/96_tmp_data_augmented_final/deep/250/1.0/sc/test.txt")
x_valid = np.array(x_valid)
y_valid = np.array(y_valid)


for n in layer_before_softmax:
    for fun in activation:
        MODEL = str(n) + "_" + str(fun)

        RESULTS_PATH = str("/flush1/raj034/DNN/100EPOCH/deep/" + str(MODEL) + "/")
        os.system("mkdir " + RESULTS_PATH)
        LOG_FILE = open(RESULTS_PATH + 'log.txt', 'a')

        logEntry(strftime("%Y-%m-%d %H:%M:%S", localtime()) + " **** traing stated " + MODEL)

        tf.reset_default_graph()

        # tf placeholder
        # value in the range of (0, 1)
        tf_x = tf.placeholder(tf.float32, [None, 96 * 250])
        tf_y = tf.placeholder(tf.float32, [None, 10])

        if(fun == "tanh"):
            act = tf.nn.tanh
        if(fun == "relu"):
            act = tf.nn.relu
        if(fun == "sigmoid"):
            act = tf.nn.sigmoid
        if(n == 32):
                # # model 1 - sniff - 32
            hidden_layer1 = tf.layers.dense(tf_x, 16384, act, name="hidden_layer1")
            hidden_layer2 = tf.layers.dense(hidden_layer1, 8192, act, name="hidden_layer2")
            hidden_layer3 = tf.layers.dense(hidden_layer2, 4096, act, name="hidden_layer3")
            hidden_layer4 = tf.layers.dense(hidden_layer3, 2048, act, name="hidden_layer4")
            hidden_layer5 = tf.layers.dense(hidden_layer4, 512, act, name="hidden_layer5")
            hidden_layer6 = tf.layers.dense(hidden_layer5, 128, act, name="hidden_layer6")
            hidden_layer7 = tf.layers.dense(hidden_layer6, 32, act, name="hidden_layer7")
            hidden_layer8 = tf.layers.dense(hidden_layer7, 10, act, name="hidden_layer8")

        if(n == 64):
            # # model 1 - sniff - 32
            hidden_layer1 = tf.layers.dense(tf_x, 16384, act, name="hidden_layer1")
            hidden_layer2 = tf.layers.dense(hidden_layer1, 8192, act, name="hidden_layer2")
            hidden_layer3 = tf.layers.dense(hidden_layer2, 4096, act, name="hidden_layer3")
            hidden_layer4 = tf.layers.dense(hidden_layer3, 1024, act, name="hidden_layer4")
            hidden_layer5 = tf.layers.dense(hidden_layer4, 256, act, name="hidden_layer5")
            hidden_layer6 = tf.layers.dense(hidden_layer5, 64, act, name="hidden_layer6")
            # hidden_layer7 = tf.layers.dense(hidden_layer6, 32, act, name="hidden_layer7")
            hidden_layer8 = tf.layers.dense(hidden_layer6, 10, act, name="hidden_layer8")

        if(n == 128):
            # # model 1 - sniff - 32
            hidden_layer1 = tf.layers.dense(tf_x, 16384, act, name="hidden_layer1")
            hidden_layer2 = tf.layers.dense(hidden_layer1, 8192, act, name="hidden_layer2")
            hidden_layer3 = tf.layers.dense(hidden_layer2, 2048, act, name="hidden_layer3")
            hidden_layer4 = tf.layers.dense(hidden_layer3, 512, act, name="hidden_layer4")
            hidden_layer5 = tf.layers.dense(hidden_layer4, 128, act, name="hidden_layer5")
            # hidden_layer6 = tf.layers.dense(hidden_layer5, 64, act, name="hidden_layer6")
            # hidden_layer7 = tf.layers.dense(hidden_layer6, 32, act, name="hidden_layer7")
            hidden_layer8 = tf.layers.dense(hidden_layer5, 10, act, name="hidden_layer8")

        # model 3 - deep
        # hidden_layer1 = tf.layers.dense(tf_x, 18000, tf.nn.tanh, name="hidden_layer1")
        # hidden_layer2 = tf.layers.dense(hidden_layer1, 9000, tf.nn.tanh, name="hidden_layer2")
        # hidden_layer3 = tf.layers.dense(hidden_layer2, 5000, tf.nn.tanh, name="hidden_layer3")
        # hidden_layer4 = tf.layers.dense(hidden_layer3, 2000, tf.nn.tanh, name="hidden_layer4")
        # hidden_layer5 = tf.layers.dense(hidden_layer4, 800, tf.nn.tanh, name="hidden_layer5")
        # hidden_layer6 = tf.layers.dense(hidden_layer5, 200, tf.nn.tanh, name="hidden_layer6")
        # hidden_layer7 = tf.layers.dense(hidden_layer6, 50, tf.nn.tanh, name="hidden_layer7")
        # hidden_layer8 = tf.layers.dense(hidden_layer7, 10, tf.nn.tanh, name="hidden_layer8")

        softmax_layer = tf.nn.softmax(hidden_layer8, name="softmax")

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=softmax_layer, labels=tf_y))
        # loss = tf.losses.mean_squared_error(labels=tf_y, predictions=softmax_layer)
        train = tf.train.AdamOptimizer(LR).minimize(loss)

        loss_graph = []
        accuracy = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=300)
            for epoch in range(EPOCH):
                loss_avg = 0
                total_batch = int(x_train.shape[0] / BATCH_SIZE)
                for i in range(total_batch):
                    _, loss_val = sess.run([train, loss], {tf_x: np.reshape(x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE], (
                        BATCH_SIZE, 96 * 250)), tf_y: np.reshape(y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE], (BATCH_SIZE, 10))})
                    loss_avg += loss_val
                _, loss_val = sess.run([train, loss], {tf_x: np.reshape(x_train[total_batch * BATCH_SIZE:], (x_train.shape[0] - total_batch * BATCH_SIZE, 96 * 250)),
                                                       tf_y: np.reshape(y_train[total_batch * BATCH_SIZE:], (x_train.shape[0] - total_batch * BATCH_SIZE, 10))})
                logEntry("EPOCH ==> " + str(epoch) + "  LOSS  ===>  " + str(loss_avg / i))
                loss_avg += loss_val
                loss_graph.append(loss_avg / (i + 1))

                np.save(RESULTS_PATH + "loss_graph.npy", np.array(loss_graph))
                plt.figure(0)
                plt.plot(np.array(loss_graph))
                plt.savefig(RESULTS_PATH + "loss_graph.png")
                os.system("mkdir " + RESULTS_PATH + str(epoch))
                saver.save(sess, RESULTS_PATH + str(epoch) + "/dnn.ckpt")

                output = sess.graph.get_tensor_by_name("softmax:0")
                predictions = []
                for i in range(x_valid.shape[0]):
                    result = sess.run([output], feed_dict={tf_x: np.reshape(x_valid[i], (1, -1))})
                    predictions.append(result[0][0])
                accuracy.append(plotConfusionMatrix(list(predictions), y_valid, RESULTS_PATH + str(epoch) + "/confusion_matrix.png", MODEL))
                np.save(RESULTS_PATH + "accuracy_graph.npy", np.array(accuracy))
                plt.figure(1)
                plt.plot(np.array(accuracy))
                plt.savefig(RESULTS_PATH + "accuracy_graph.png")
            sess.close()
