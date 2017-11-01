import tensorflow as tf
import argparse
import numpy as np
from skimage.measure import structural_similarity as ssim
import tensorflow.contrib.graph_editor as ge
from tensorflow.python.tools import freeze_graph
import os
import math


quant = "quantization_no"
breath = "sniff"
LOG_FILE = open("/flush1/raj034/DNN/model/test_cases/" + breath + "/" + quant + "/log.txt", 'a')


def logEntry(TMP_STRING):
    LOG_FILE.write(str(TMP_STRING))
    LOG_FILE.write("\n")
    LOG_FILE.flush()
    print(TMP_STRING)


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def svd_compress_gs(mat, k):
    """Given a matrix representing a grayscale image, compress
    it by taking the largest k elements from its singular values"""
    U, singular_vals, V = np.linalg.svd(mat)
    rank = len(singular_vals)
    print("Image rank %r" % rank)
    if k > rank:
        print("k is larger than rank of image %r" % rank)
        return mat
    # take columns less than k from U
    U_p = U[:, :k]
    # take rows less than k from V
    V_p = V[:k, :]
    # build the new S matrix with top k diagnal elements
    S_p = np.zeros((k, k), mat.dtype)
    for i in range(k):
        S_p[i][i] = singular_vals[i]
    print("U_p shape {0}, S_p shape {1}, V_p shape {2}".format(
        U_p.shape, S_p.shape, V_p.shape))
    compressed = np.dot(np.dot(U_p, S_p), V_p)
    ss = ssim(mat, compressed,
              dynamic_range=compressed.max() - compressed.min())
    print("Strucural similarity: %r" % ss)
    return U_p, S_p, V_p, ss


def build_pb(file, lvl, svd_num):
    tf.reset_default_graph()
    graph = load_graph(file)
    logEntry(svd_num)
    with tf.Session(graph=graph) as sess:
        count = 0
        pre_layer = ['Placeholder', 'hidden_layer1', 'hidden_layer2', 'hidden_layer3', 'hidden_layer4', 'hidden_layer5', 'hidden_layer6']
        # svd_num = [40, 8, 8, 8, 8, 8, 8]
        for layer in ['hidden_layer1', 'hidden_layer2', 'hidden_layer3', 'hidden_layer4', 'hidden_layer5', 'hidden_layer6', 'hidden_layer7']:
            if(count == 0):
                layer_input = graph.get_tensor_by_name('prefix/' + pre_layer[count] + ':0')
            else:
                layer_input = graph.get_tensor_by_name('prefix/' + pre_layer[count] + '/Tanh:0')
            W = graph.get_tensor_by_name('prefix/' + layer + '/kernel:0')
            matmul = graph.get_tensor_by_name('prefix/' + layer + '/MatMul:0')
            add = graph.get_tensor_by_name('prefix/' + layer + '/BiasAdd:0')

            W1 = W.eval()
            print(W1.shape)
            u_1, s_1, v_1, ss = svd_compress_gs(W1, svd_num[count])
            logEntry(str(lvl) + " ==> layer ==> " + str(count) + " ssim   ==> " + str(ss))
            u1 = tf.matmul(layer_input, u_1, name="prefix/" + layer + "/u1")
            s1 = tf.matmul(u1, s_1, name="prefix/" + layer + "/s1")
            v1 = tf.matmul(s1, v_1, name="prefix/" + layer + "/v1")
            ge.connect(ge.sgv(v1.op), ge.sgv(add.op).remap_inputs([0]))

            count += 1

        a = tf.Variable(5, name="dummy")
        sess.run(tf.variables_initializer([a]))
        saver = tf.train.Saver()

        os.system("mkdir /flush1/raj034/DNN/model/test_cases/" + breath + "/" + quant + "/fact_" + str(lvl))
        os.system("mkdir /flush1/raj034/DNN/model/test_cases/" + breath + "/" + quant + "/fact_" + str(lvl) + "/pb")

        LOGDIR = "/flush1/raj034/DNN/model/test_cases/" + breath + "/" + quant + "/fact_" + str(lvl) + "/LOG"
        train_writer = tf.summary.FileWriter(LOGDIR)
        train_writer.add_graph(sess.graph)
        train_writer.close()

        tf.train.write_graph(sess.graph_def, "/flush1/raj034/DNN/model/test_cases/" + breath + "/" + quant + '/fact_' +
                             str(lvl) + '/pb/', "DNN_" + breath + "_" + quant + "_fact_" + str(lvl) + ".pbtxt")
        saver.save(sess, save_path="/flush1/raj034/DNN/model/test_cases/" + breath + "/" + quant + '/fact_' + str(lvl) + "/model.ckpt")

        input_graph_path = "/flush1/raj034/DNN/model/test_cases/" + breath + "/" + quant + '/fact_' + \
            str(lvl) + '/pb/' + "DNN_" + breath + "_" + quant + "_fact_" + str(lvl) + ".pbtxt"

        checkpoint_path = "/flush1/raj034/DNN/model/test_cases/" + breath + "/" + quant + '/fact_' + str(lvl) + "/model.ckpt"
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        output_frozen_graph_name = "/flush1/raj034/DNN/model/test_cases/" + breath + "/" + quant + '/fact_' + \
            str(lvl) + '/pb/' + "DNN_" + breath + "_" + quant + "_fact_" + str(lvl) + ".pb"

        logEntry("Start Freezing the graph")

        freeze_graph.freeze_graph(input_graph_path, input_saver="",
                                  input_binary=False, input_checkpoint=checkpoint_path,
                                  output_node_names="prefix/softmax", restore_op_name="save/restore_all",
                                  filename_tensor_name="save/Const:0",
                                  output_graph=output_frozen_graph_name, clear_devices=True, initializer_nodes="")

        logEntry("End Freezing the graph")
        sess.close()


if __name__ == '__main__':
    logEntry("started")

    ssim_avg = [90, 80, 70, 60, 50, 40, 30, 20, 10]

    for lvl in ssim_avg:
        svd_num = [2000, 1500, 1000, 500, 200, 50, 10]
        svd_num_2 = [math.ceil(x * lvl / 100) for x in svd_num]
        logEntry(lvl)
        logEntry(svd_num)
        # load the orignal graph
        # graph = load_graph("../500/tmp/har.pb")

        build_pb("../model/pb_files/dnn_sniff.pb", lvl, svd_num_2)

    print("done")
