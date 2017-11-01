# http://slazebni.cs.illinois.edu/spring17/lec06_compression.pdf
# https://dato.ml/tensorflow-mobile-graph-optimization/
# https://petewarden.com/2016/09/27/tensorflow-for-mobile-poets/
# python ./tensorflow/tensorflow/tools/quantization/quantize_graph.py --input=frozen_har.pb --output=rounded_graph.pb --output_node_names=y_ --mode=weights_rounded
# Inside the GIT folder
# https://stackoverflow.com/questions/45481000/inceptionv3-retrain-py-when-frozen-graph-for-inference-different-accuracy-betw
# https://github.com/llSourcell/tensorflow_image_classifier/blob/master/src/label_image.py
# https://medium.com/@erikhallstrm/using-the-tensorflow-lstm-api-3-7-5f2b97ca6b73

import csv
import numpy as np
import pickle
from scipy import stats
from collections import deque

from sklearn import metrics
from sklearn.model_selection import train_test_split
from time import localtime, strftime
import pathlib
import os
import glob
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import tensorflow as tf

from tensorflow.python.tools import freeze_graph
from tensorflow.python.platform import gfile

# Plotting Paramters
#%matplotlib inline
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8

quant = "quantization_no"
breath = "sniff"
LOG_FILE = open("/flush1/raj034/DNN/model/test_cases/" + breath + "/" + quant + "/log.txt", 'a')


def logEntry(TMP_STRING):
    LOG_FILE.write(str(TMP_STRING))
    LOG_FILE.write("\n")
    LOG_FILE.flush()
    print(TMP_STRING)


##################### Start of the data proprocessing function #####################


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

################## End of stored data preprocess function ######################


################# End of the LSTM Model Parameters function ####################


def plotConfusionMatrix(predictions, y, i, RESULTS_PATH,  FILENAME, TITLE):
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
    fig.savefig(RESULTS_PATH + FILENAME)
    plt.clf()
    plt.close('all')

    return

#################################### Inference Fucntion ################################


def Inference(filePath, X_valid):
    tf.reset_default_graph()
    with tf.gfile.FastGFile(filePath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # session = tf.Session()
    # sess = tf.Session()
    # config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
    with tf.Session(config=config) as sess:

        X = sess.graph.get_tensor_by_name("prefix/Placeholder:0")
        pred_softmax = sess.graph.get_tensor_by_name("prefix/softmax:0")
        # predict_fc = sess.graph.get_tensor_by_name("prefix/rnn/multi_rnn_cell_29/cell_1/dropout/mul:0")

        predictions = []

        logEntry(strftime("%Y-%m-%d %H:%M:%S", localtime()) + str(X_valid.shape))
        i = 1

        vect_out = []
        for x in range(0, X_valid.shape[0]):
            # print(X_valid[x].shape)
            # tmp = np.expand_dims(X_valid[x], axis=0)
            tmp = np.reshape(X_valid[x], (1, -1))
            # print(tmp.shape)
            # if(i % 100 == 0):
            #     logEntry(strftime("%Y-%m-%d %H:%M:%S", localtime()) + " " + str(i) + " START")
            tmp, vect_128 = sess.run([pred_softmax, pred_softmax], feed_dict={X: tmp})
            predictions.append(tmp[0])
            vect_out.append(vect_128[0])
            i = i + 1
        sess.close()
    return predictions, vect_128

#################################### Start of main thread #############################

tests = os.listdir("/flush1/raj034/DNN/model/test_cases/" + breath + "/" + quant + "/")
tests.sort()
tests.remove('log.txt')
logEntry(tests)

X_valid, y_valid = buildTestSet("/flush1/sen040/96_tmp_data_augmented_final/sniff/30/1.0/sc/test.txt")
# X_valid, y_valid = buildTestSet("/flush1/sen040/96_tmp_data_augmented_final/deep/250/1.0/sc/test.txt")

logEntry(np.ndim(X_valid))
logEntry(np.ndim(y_valid))

for cases in tests:
    path = "/flush1/raj034/DNN/model/test_cases/" + breath + "/" + quant + "/" + str(cases)
    logEntry(path)
    logEntry(strftime("%Y-%m-%d %H:%M:%S", localtime()) + " **** Loading Models")

    predictions_frozen = []
    predictions_frozen, vect_out = Inference(path + "/pb/DNN_" + breath + "_" + quant + "_" + str(cases) + ".pb", X_valid)
    os.system("zip -r " + path + "/pb/compressed.zip " + path + "/pb/DNN_" + breath + "_" + quant + "_" + str(cases) + ".pb")
    logEntry(os.system("ls -l " + path + "/pb/compressed.zip"))

    # save vetors and class
    # print(np.array(vect_out).shape)
    # np.save("./model/tmp/npy/class.npy", np.array(y_valid))
    # np.save("./model/tmp/npy/vect_128.npy", np.array(vect_out))

    logEntry(strftime("%Y-%m-%d %H:%M:%S", localtime()) + " **** Done Loading Models")

    # print(len(predictions_frozen))
    # print(predictions_frozen[0])

    #######################################################################################

    plotConfusionMatrix(list(predictions_frozen), y_valid, 100, path + "/", "confusion_matrix.png", "RNN - " + breath + "_" + quant)

print("Fianlly done")
