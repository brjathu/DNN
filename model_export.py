from sklearn import metrics
from sklearn.model_selection import train_test_split
from time import localtime, strftime
import pathlib
import os
import glob
import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import tensorflow as tf

from tensorflow.python.tools import freeze_graph


MODEL = 1
print("Start")

sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph('/flush1/raj034/DNN/model/' + str(MODEL) + '/499/dnn.ckpt.meta')
new_saver.restore(sess, '/flush1/raj034/DNN/model/' + str(MODEL) + '/499/dnn.ckpt')

print("Model Loading Done")

tf.get_default_graph().as_graph_def()

# pred_softmax = tf.nn.softmax(pred_Y, name="y_")
X = sess.graph.get_tensor_by_name("Placeholder:0")
pred_softmax = sess.graph.get_tensor_by_name("softmax:0")

# tf.train.write_graph(sess.graph_def, RESULTS_PATH+'/812/tmp/', 'tfdroid.pbtxt')

print("Writing the graph")

os.system("mkdir model/" + str(MODEL) + "/pb")
tf.train.write_graph(sess.graph_def, "model/" + str(MODEL) + "/pb/", "model.pbtxt")
new_saver.save(sess, save_path="model/" + str(MODEL) + "/pb/model.ckpt")

input_graph_path = "model/" + str(MODEL) + "/pb/model.pbtxt"
checkpoint_path = "model/" + str(MODEL) + "/pb/model.ckpt"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = "model/" + str(MODEL) + "/pb/model.pb"

print("Start Freezing the graph")

freeze_graph.freeze_graph(input_graph_path, input_saver="",
                          input_binary=False, input_checkpoint=checkpoint_path,
                          output_node_names="softmax", restore_op_name="save/restore_all",
                          filename_tensor_name="save/Const:0",
                          output_graph=output_frozen_graph_name, clear_devices=True, initializer_nodes="")

print("End Freezing the graph")
