import tensorflow as tf
from tensorflow.python.platform import gfile


MODEL = 4
print("Start")
sess = tf.InteractiveSession()
new_saver = tf.train.import_meta_graph("model/" + str(MODEL) + "/ckpt/dnn.ckpt.meta")
new_saver.restore(sess, "model/" + str(MODEL) + "/ckpt/dnn.ckpt")
print("Model Loading Done")

tf.get_default_graph().as_graph_def()

for op in sess.graph.get_operations():
    print(op.name)


LOGDIR='LOG'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
train_writer.close()

print("saved")