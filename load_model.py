import tensorflow as tf
import argparse
import numpy as np
from skimage.measure import structural_similarity as ssim
import tensorflow.contrib.graph_editor as ge
from tensorflow.python.tools import freeze_graph


MODEL = 1


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
    return U_p, S_p, V_p


if __name__ == '__main__':
    print("started")
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="model/" + str(MODEL) + "/pb/model.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()
    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)

    with tf.Session(graph=graph) as sess:
        count = 0
        pre_layer = ['Placeholder', 'hidden_layer1', 'hidden_layer2', 'hidden_layer3', 'hidden_layer4', 'hidden_layer5', 'hidden_layer6']
        svd_num = [40, 8, 8, 8, 8, 8, 8]
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
            u_1, s_1, v_1 = svd_compress_gs(W1, svd_num[count])
            u1 = tf.matmul(layer_input, u_1, name="prefix/" + layer + "/u1")
            s1 = tf.matmul(u1, s_1, name="prefix/" + layer + "/s1")
            v1 = tf.matmul(s1, v_1, name="prefix/" + layer + "/v1")
            ge.connect(ge.sgv(v1.op), ge.sgv(add.op).remap_inputs([0]))

            count += 1

        # # at output fc layer
        # W1_u = W1_u.eval()
        # u_u,s_u,v_u = svd_compress_gs(W1_u,5)
        # u1_u = tf.matmul(cell29 , u_u, name="prefix/u1_u")
        # s1_u = tf.matmul(u1_u, s_u, name="prefix/s1_u")
        # v1_u = tf.matmul(s1_u , v_u, name="prefix/v1_u")
        # ge.connect(ge.sgv(v1_u.op), ge.sgv(add_u.op).remap_inputs([0]))

        a = tf.Variable(5, name="dummy")
        sess.run(tf.variables_initializer([a]))
        saver = tf.train.Saver()

        save_path = saver.save(sess, "model/" + str(MODEL) + "/test.ckpt")

        LOGDIR = 'LOG_test'
        train_writer = tf.summary.FileWriter(LOGDIR)
        train_writer.add_graph(sess.graph)
        train_writer.close()

    print("done")
