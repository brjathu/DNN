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


# parameters
LR = 0.0005
EPOCH = 100
BATCH_SIZE = 200
RESULTS_PATH = "./model/1/"
LOG_FILE=open(RESULTS_PATH+'log.txt', 'a')


def logEntry(TMP_STRING):
    LOG_FILE.write(TMP_STRING)
    LOG_FILE.write("\n")
    LOG_FILE.flush()
    print(TMP_STRING)


def buildTestSet(fileName):
    logEntry(strftime("%Y-%m-%d %H:%M:%S", localtime())+" **** Building Dataset")
    X=[]
    Y=[]
    with open(fileName) as file:
        for i in file:
            c=i.rstrip()
            y=c[-30:]
            x=c[:-30]
            
            x_list=eval(x)
            y=eval(y)
            
            X.append(x_list)
            Y.append(y)
            
    X = np.asarray(X)
    logEntry(strftime("%Y-%m-%d %H:%M:%S", localtime())+" **** Done Building Dataset")
    return X,Y            


tf.reset_default_graph()

# tf placeholder
tf_x = tf.placeholder(tf.float32, [None, 96*30])    # value in the range of (0, 1)
tf_y = tf.placeholder(tf.float32, [None, 10])

# layers
hidden_layer1 = tf.layers.dense(tf_x, 2000, tf.nn.relu)
hidden_layer2 = tf.layers.dense(hidden_layer1, 1000, tf.nn.relu)
hidden_layer3 = tf.layers.dense(hidden_layer2, 500, tf.nn.relu)
hidden_layer4 = tf.layers.dense(hidden_layer3, 250, tf.nn.relu)
hidden_layer5 = tf.layers.dense(hidden_layer4, 125, tf.nn.relu)
hidden_layer6 = tf.layers.dense(hidden_layer5, 50, tf.nn.relu)
hidden_layer7 = tf.layers.dense(hidden_layer6, 10, tf.nn.relu)

softmax_layer = tf.nn.softmax(hidden_layer7)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = softmax_layer, labels = tf_y)) 
# loss = tf.losses.mean_squared_error(labels=tf_y, predictions=softmax_layer)
train = tf.train.AdamOptimizer(LR).minimize(loss)



# # prepare data
# x_valid, y_valid = buildTestSet("./data/data_test_balanced.txt")
# x_valid = np.array(x_valid)
# y_valid = np.array(y_valid)

# #preprocessing 
# X = []
# Y =[]
# for i in range(x_valid.shape[0]):
# 	X.append( np.reshape(x_valid[i],(1,-1))[0])
# 	Y.append( np.reshape(y_valid[i],(1,-1))[0])

# np.save("./data/x_valid.npy", np.array(X))
# np.save("./data/y_valid.npy", np.array(Y))

# load data
x = np.load("./data/x_valid.npy")
y = np.load("./data/y_valid.npy")
print(x.shape)
print(y.shape)

ratio = int(x.shape[0]*80/100)
x_train = x[:ratio,:]
y_train = y[:ratio,:]

x_test = x[ratio:,:]
y_test = y[ratio:,:]

# classifier = tf.contrib.learn.DNNClassifier(hidden_units=[1000, 500, 100, 10],
# 								n_classes=2,
# 								feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(x_train),
# 								optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.05))

loss_graph = []
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	for epoch in range(EPOCH):
		for i in range(int(x_train.shape[0]/BATCH_SIZE)):
			_, loss_val = sess.run([train, loss], {tf_x: np.reshape(x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE],(BATCH_SIZE,2880)) , tf_y:np.reshape(y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE],(BATCH_SIZE,10))})
			

		print(loss_val)
		loss_graph.append(loss_val)
		np.save("./model/1/train.npy",np.array(loss_graph))
		plt.plot(np.array(loss_graph))
		plt.savefig("./model/1/loss_graph.png")
		# plt.show()

		saver.save(sess, "model/1/ckpt/dnn.ckpt")

