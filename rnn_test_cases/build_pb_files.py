import tensorflow as tf
import argparse
import numpy as np
from skimage.measure import structural_similarity as ssim
import tensorflow.contrib.graph_editor as ge
from tensorflow.python.tools import freeze_graph
import os


quant = "quantization_no"
breath = "deep-32-2"
LOG_FILE = open("/flush1/raj034/RNN/model/test_cases/" + breath + "/" + quant + "/log.txt", 'a')


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


if __name__ == '__main__':
    logEntry("started")

    fact_levels = [32, 30, 25, 20, 15, 10, 5]

    for lvl in fact_levels:
        logEntry(lvl)
        # load the orignal graph
        graph = load_graph("../model/pb_files/rnn-deep-250-32-2.pb")

        W1 = graph.get_tensor_by_name('prefix/w_in:0')
        matmul = graph.get_tensor_by_name('prefix/MatMul:0')
        bias = graph.get_tensor_by_name('prefix/b_in:0')
        add = graph.get_tensor_by_name('prefix/add:0')
        reshape = graph.get_tensor_by_name('prefix/Reshape:0')

        # #remove all conncetions from matmul
        ge.detach(ge.sgv(matmul.op))

        with tf.Session(graph=graph) as sess:
            os.system("mkdir /flush1/raj034/RNN/model/test_cases/" + breath + "/" + quant + "/fact_" + str(lvl))

            # for op in sess.graph.get_operations():
            #     print(op.name)

            W = W1.eval()
            u, s, v, ss = svd_compress_gs(W, lvl)
            logEntry("structural_similarity == > " + str(ss))
            u1 = tf.matmul(reshape, u, name="prefix/u1")
            s1 = tf.matmul(u1, s, name="prefix/s1")
            v1 = tf.matmul(s1, v, name="prefix/v1")
            ge.connect(ge.sgv(v1.op), ge.sgv(add.op).remap_inputs([0]))

            sess.run(tf.variables_initializer([tf.Variable(5, name="dummy" + str(lvl))]))
            saver = tf.train.Saver()

            # save log for tensorboad
            LOGDIR = "/flush1/raj034/RNN/model/test_cases/" + breath + "/" + quant + '/fact_' + str(lvl) + '/LOG'
            train_writer = tf.summary.FileWriter(LOGDIR)
            train_writer.add_graph(sess.graph)
            train_writer.close()

            # save the freezed model
            os.system("mkdir /flush1/raj034/RNN/model/test_cases/" + breath + "/" + quant + "/fact_" + str(lvl) + "/pb")
            tf.train.write_graph(sess.graph_def, "/flush1/raj034/RNN/model/test_cases/" + breath + "/" + quant + '/fact_' +
                                 str(lvl) + '/pb/', "RNN_" + breath + "_" + quant + "_fact_" + str(lvl) + ".pbtxt")
            saver.save(sess, save_path="/flush1/raj034/RNN/model/test_cases/" + breath + "/" + quant + '/fact_' + str(lvl) + "/model.ckpt")

            input_graph_path = "/flush1/raj034/RNN/model/test_cases/" + breath + "/" + quant + '/fact_' + \
                str(lvl) + '/pb/' + "RNN_" + breath + "_" + quant + "_fact_" + str(lvl) + ".pbtxt"

            checkpoint_path = "/flush1/raj034/RNN/model/test_cases/" + breath + "/" + quant + '/fact_' + str(lvl) + "/model.ckpt"
            restore_op_name = "save/restore_all"
            filename_tensor_name = "save/Const:0"
            output_frozen_graph_name = "/flush1/raj034/RNN/model/test_cases/" + breath + "/" + quant + '/fact_' + \
                str(lvl) + '/pb/' + "RNN_" + breath + "_" + quant + "_fact_" + str(lvl) + ".pb"

            logEntry("Start Freezing the graph")

            freeze_graph.freeze_graph(input_graph_path, input_saver="",
                                      input_binary=False, input_checkpoint=checkpoint_path,
                                      output_node_names="prefix/y_", restore_op_name="save/restore_all",
                                      filename_tensor_name="save/Const:0",
                                      output_graph=output_frozen_graph_name, clear_devices=True, initializer_nodes="")

            logEntry("End Freezing the graph")

            sess.close()
    print("done")
