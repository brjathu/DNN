# Copyright (c) 2017
#
# Jathushan.Rajasegaran@gmail.com


import os
import tensorflow as tf
from time import localtime, strftime
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns


tf.set_random_seed(1)
tf.logging.set_verbosity(tf.logging.ERROR)


# parameters
LR = 0.0005
EPOCH = 100
BATCH_SIZE = 200
MODEL = 5
RESULTS_PATH = "./model/1/"
LOG_FILE = open(RESULTS_PATH + 'log.txt', 'a')


def logEntry(TMP_STRING):
    LOG_FILE.write(TMP_STRING)
    LOG_FILE.write("\n")
    LOG_FILE.flush()
    print(TMP_STRING)


def buildTestSet(fileName):
    logEntry(strftime("%Y-%m-%d %H:%M:%S", localtime()) + " **** Building Dataset")
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
    logEntry(strftime("%Y-%m-%d %H:%M:%S", localtime()) + " **** Done Building Dataset")
    return X, Y


def plotConfusionMatrix(predictions, y, i, FILENAME, TITLE):
    LABELS = ['User 1', 'User 2', 'User 3', 'User 4', 'User 5', 'User 6', 'User 7', 'User 8', 'User 9', 'User 10']
    max_test = np.argmax(y, axis=1)
    max_predictions = np.argmax(predictions, axis=1)
    confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)
    fig = plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
    plt.title("Confusion matrix: " + TITLE)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig("./model/" + str(MODEL) + "/" + FILENAME)
    plt.clf()
    plt.close('all')

    return


print("Start")
sess = tf.InteractiveSession()
new_saver = tf.train.import_meta_graph("model/" + str(MODEL) + "/ckpt/dnn.ckpt.meta")
new_saver.restore(sess, "model/" + str(MODEL) + "/ckpt/dnn.ckpt")
print("Model Loading Done")

tf.get_default_graph().as_graph_def()

input_ph = sess.graph.get_tensor_by_name("Placeholder:0")
output = sess.graph.get_tensor_by_name("softmax:0")


LOGDIR = 'LOG'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)

for op in sess.graph.get_operations():
    print(op.name)


# load data
x = np.load("./data/x_train.npy")
y = np.load("./data/y_train.npy")
print(x.shape)
print(y.shape)

ratio = int(x.shape[0] * 80 / 100)
x_train = x[:ratio, :]
y_train = y[:ratio, :]

x_test = x[ratio:, :]
y_test = y[ratio:, :]
print(y_test)
predictions = []
for i in range(x_test.shape[0]):
    result = sess.run([output], feed_dict={input_ph: np.reshape(x_test[i], (1, -1))})
    predictions.append(result[0][0])
    print(i)

print(np.array(predictions).shape)
plotConfusionMatrix(list(predictions), y_test, 100, "confusion_matrix.png", "model1")
