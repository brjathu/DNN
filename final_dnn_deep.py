from sklearn import metrics
from sklearn.model_selection import train_test_split
from time import localtime, strftime
import pathlib
import os
import glob
import matplotlib
matplotlib.use('Agg')
import math

import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import tensorflow as tf

from tensorflow.python.tools import freeze_graph
import argparse
import numpy as np
from skimage.measure import structural_similarity as ssim
import tensorflow.contrib.graph_editor as ge


def logEntry(TMP_STRING):
    LOG_FILE.write(str(TMP_STRING))
    LOG_FILE.write("\n")
    LOG_FILE.flush()
    print(TMP_STRING)


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


def build_pb_fact(pb_location, main_location, breath, quant, lvl):
    graph = load_graph(pb_location)

    with tf.Session(graph=graph) as sess:
        count = 0
        pre_layer = ['Placeholder', 'hidden_layer1', 'hidden_layer2', 'hidden_layer3', 'hidden_layer4', 'hidden_layer5', 'hidden_layer6']
        svd_num = [2048, 1024, 512, 256, 128, 64, 32]
        for layer in ['hidden_layer1', 'hidden_layer2', 'hidden_layer3', 'hidden_layer4', 'hidden_layer5', 'hidden_layer6', 'hidden_layer7']:
            if(count == 0):
                layer_input = graph.get_tensor_by_name('prefix/' + pre_layer[count] + ':0')
            else:
                layer_input = graph.get_tensor_by_name('prefix/' + pre_layer[count] + '/Tanh:0')
            W = graph.get_tensor_by_name('prefix/' + layer + '/kernel:0')
            matmul = graph.get_tensor_by_name('prefix/' + layer + '/MatMul:0')
            add = graph.get_tensor_by_name('prefix/' + layer + '/BiasAdd:0')
            ge.detach(ge.sgv(matmul.op))

            W1 = W.eval()
            print(W1.shape)
            u_1, s_1, v_1, ss = svd_compress_gs(W1, math.ceil(svd_num[count] * lvl / 100))
            logEntry(str(lvl) + " ==> layer ==> " + str(count) + " ssim   ==> " + str(ss))
            u1 = tf.matmul(layer_input, u_1, name="prefix/" + layer + "/u1")
            s1 = tf.matmul(u1, s_1, name="prefix/" + layer + "/s1")
            v1 = tf.matmul(s1, v_1, name="prefix/" + layer + "/v1")
            ge.connect(ge.sgv(v1.op), ge.sgv(add.op).remap_inputs([0]))

            count += 1

        sess.run(tf.variables_initializer([tf.Variable(5, name="dummy" + str(lvl))]))
        saver = tf.train.Saver()

        # save log for tensorboad
        LOGDIR = main_location + '/LOG'
        train_writer = tf.summary.FileWriter(LOGDIR)
        train_writer.add_graph(sess.graph)
        train_writer.close()

        # save the freezed model
        os.system("mkdir " + main_location + "pb")
        tf.train.write_graph(sess.graph_def, main_location + 'pb/', "DNN_" + breath + "_" + quant + "_fact_" + str(lvl) + ".pbtxt")
        saver.save(sess, save_path=main_location + "model.ckpt")

        input_graph_path = main_location + '/pb/' + "DNN_" + breath + "_" + quant + "_fact_" + str(lvl) + ".pbtxt"

        checkpoint_path = main_location + "model.ckpt"
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        output_frozen_graph_name = main_location + 'pb/' + "DNN_" + breath + "_" + quant + "_fact_" + str(lvl) + ".pb"

        logEntry("Start Freezing the graph")

        freeze_graph.freeze_graph(input_graph_path, input_saver="",
                                  input_binary=False, input_checkpoint=checkpoint_path,
                                  output_node_names="prefix/softmax", restore_op_name="save/restore_all",
                                  filename_tensor_name="save/Const:0",
                                  output_graph=output_frozen_graph_name, clear_devices=True, initializer_nodes="")

        logEntry("End Freezing the graph")

        sess.close()


def export_model(sess_loc, pb_loc):
    sess = tf.InteractiveSession()
    # sess.reset_default_graph()
    new_saver = tf.train.import_meta_graph(sess_loc + "dnn.ckpt.meta")
    new_saver.restore(sess, sess_loc + "dnn.ckpt")
    logEntry("Model Loading Done" + str(sess_loc))
    tf.get_default_graph().as_graph_def()
    # # pred_softmax = tf.nn.softmax(pred_Y, name="y_")
    # X = sess.graph.get_tensor_by_name("Placeholder:0")
    # pred_softmax = sess.graph.get_tensor_by_name("softmax:0")
    logEntry("Writing the graph")
    # os.system("mkdir " + str(sess_loc) + "/pb")
    tf.train.write_graph(sess.graph_def, pb_loc, "model.pbtxt")
    new_saver.save(sess, save_path=pb_loc + "model.ckpt")
    input_graph_path = pb_loc + "model.pbtxt"
    checkpoint_path = pb_loc + "model.ckpt"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = pb_loc + "model.pb"
    logEntry("Start Freezing the graph")
    freeze_graph.freeze_graph(input_graph_path, input_saver="",
                              input_binary=False, input_checkpoint=checkpoint_path,
                              output_node_names="softmax", restore_op_name="save/restore_all",
                              filename_tensor_name="save/Const:0",
                              output_graph=output_frozen_graph_name, clear_devices=True, initializer_nodes="")
    logEntry("End Freezing the graph")
    sess.close()


def buildTestSet(fileName):
    # logEntry(strftime("%Y-%m-%d %H:%M:%S", localtime()) + " **** Building Dataset")
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
    # logEntry(strftime("%Y-%m-%d %H:%M:%S", localtime()) + " **** Done Building Dataset")
    return X, Y

################## End of stored data preprocess function ######################


################# End of the LSTM Model Parameters function ####################


def plotConfusionMatrix(predictions, y, i, RESULTS_PATH, FILENAME, TITLE):
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

        for op in sess.graph.get_operations():
            print(op.name)

        X = sess.graph.get_tensor_by_name("prefix/Placeholder:0")
        pred_softmax = sess.graph.get_tensor_by_name("prefix/softmax:0")
        # predict_fc = sess.graph.get_tensor_by_name("prefix/rnn/multi_rnn_cell_29/cell_1/dropout/mul:0")
        predict_fc = sess.graph.get_tensor_by_name("prefix/hidden_layer7/Tanh:0")

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
            tmp, d_vect = sess.run([pred_softmax, predict_fc], feed_dict={X: tmp})
            predictions.append(tmp[0])
            vect_out.append(d_vect[0])
            i = i + 1
        sess.close()
    return predictions, vect_out


def Inference_2(filePath, X_valid):
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

        X = sess.graph.get_tensor_by_name("Placeholder:0")
        pred_softmax = sess.graph.get_tensor_by_name("softmax:0")
        # predict_fc = sess.graph.get_tensor_by_name("prefix/rnn/multi_rnn_cell_29/cell_1/dropout/mul:0")
        predict_fc = sess.graph.get_tensor_by_name("hidden_layer7/Tanh:0")

        predictions = []

        logEntry(strftime("%Y-%m-%d %H:%M:%S", localtime()) + str(X_valid.shape))
        i = 1

        vect_out = []
        for x in range(0, X_valid.shape[0]):
            # print(X_valid[x].shape)
            tmp = np.expand_dims(X_valid[x], axis=0)
            # tmp = np.reshape(X_valid[x], (1, 96, 30))
            # print(tmp.shape)
            # if(i % 100 == 0):
            #     logEntry(strftime("%Y-%m-%d %H:%M:%S", localtime()) + " " + str(i) + " START")
            tmp, d_vect = sess.run([pred_softmax, predict_fc], feed_dict={X: tmp})
            predictions.append(tmp[0])
            vect_out.append(d_vect[0])
            i = i + 1
        sess.close()
    return predictions, vect_out


# x_test, y_test = buildTestSet("/flush1/sen040/96_tmp_data_augmented_final/sniff/30/1.0/sc/test.txt")
x_test, y_test = buildTestSet("/flush1/sen040/96_tmp_data_augmented_final/deep/250/1.0/sc/test.txt")
x_test = np.array(x_test)
y_test = np.array(y_test)

# prepare data
# x_train, y_train = buildTestSet("/flush1/sen040/96_tmp_data_augmented_final/sniff/30/1.0/sc/train.txt")
x_train, y_train = buildTestSet("/flush1/sen040/96_tmp_data_augmented_final/deep/250/1.0/sc/train.txt")
x_train = np.array(x_train)
y_train = np.array(y_train)


main_dir = "/flush1/raj034/DNN/100EPOCH/deep/"

hidden_unit_folder = [32]
for h_folder in hidden_unit_folder:
    for layer in [2]:
        new_path = main_dir + str(h_folder) + "_tanh/99/"
        pb_location = "/flush1/raj034/DNN/final/dnn/deep/"
        model = str(h_folder) + "_tanh"
        os.system("mkdir " + pb_location + model + str("/"))
        LOG_FILE = open(pb_location + model + '/log.txt', 'a')
        export_model(new_path, pb_location + model + str("/"))
        logEntry(model)

        for breath in ["breath_deep"]:
            for quant in ["quant_no", "quant_yes"]:
                os.system("mkdir " + pb_location + model + "/" + quant)
                if(quant == "quant_yes"):
                    # 256
                    os.system("mkdir " + pb_location + model + "/" + quant + "/original/")
                    os.system("mkdir " + pb_location + model + "/" + quant + "/original/pb")
                    main_location = pb_location + model + "/" + quant + "/original/"
                    os.system("python converge_weights.py --min_n_weights 256 --file " + pb_location + model +
                              "/model.pb" + " --save_location " + main_location + "pb/quantized_original.pb")

                    # accuracy test
                    predictions_frozen = []
                    LOG_FILE = open(main_location + 'log.txt', 'a')
                    predictions_frozen, vect_out = Inference_2(main_location + "/pb/quantized_original.pb", x_test)
                    os.system("zip -r " + main_location + "/pb/compressed.zip " + main_location + "/pb/quantized_original.pb")
                    logEntry(os.system("ls -l " + main_location + "/pb/compressed.zip"))
                    plotConfusionMatrix(list(predictions_frozen), y_test, 100, main_location, "confusion_matrix.png", "DNN - " + breath + "_" + quant)

                    predictions_frozen, vect_out = Inference_2(main_location + "/pb/quantized_original.pb", x_train)
                    np.save(main_location + "d-vector.npy", vect_out)

                    predictions_frozen, vect_out = Inference_2(main_location + "/pb/quantized_original.pb", x_test)
                    np.save(main_location + "d-vector-test.npy", vect_out)

                for lvl in [100, 80, 60, 40, 20, 10, 5]:

                    if(quant == "quant_yes"):
                        # lvl_x = min(math.ceil(hid*lvl / 100),hid)

                        location = pb_location + model + "/quant_no/fact_" + str(lvl) + "/pb/DNN_" + breath + "_quant_no_fact_" + str(lvl) + ".pb"
                        os.system("mkdir " + pb_location + model + "/" + quant)
                        os.system("mkdir " + pb_location + model + "/" + quant + "/fact_" + str(lvl))
                        os.system("mkdir " + pb_location + model + "/" + quant + "/fact_" + str(lvl) + "/pb/")
                        main_location = pb_location + model + "/" + quant + "/fact_" + str(lvl) + "/"
                        LOG_FILE = open(main_location + 'log.txt', 'a')
                        os.system("python converge_weights.py --file " + location + " --save_location " +
                                  main_location + "/pb/DNN_" + breath + "_" + quant + "_fact_" + str(lvl) + ".pb")

                        # accuracy test
                        predictions_frozen = []
                        predictions_frozen, vect_out = Inference(main_location + "/pb/DNN_" + breath + "_" +
                                                                 quant + "_fact_" + str(lvl) + ".pb", x_test)
                        os.system("zip -r " + main_location + "/pb/compressed.zip " + main_location +
                                  "/pb/DNN_" + breath + "_" + quant + "_fact_" + str(lvl) + ".pb")
                        logEntry(os.system("ls -l " + main_location + "/pb/compressed.zip"))
                        plotConfusionMatrix(list(predictions_frozen), y_test, 100, main_location,
                                            "confusion_matrix.png", "DNN - " + breath + "_" + quant)

                        predictions_frozen, vect_out = Inference(main_location + "/pb/DNN_" + breath + "_" +
                                                                 quant + "_fact_" + str(lvl) + ".pb", x_train)
                        np.save(main_location + "d-vector.npy", vect_out)

                        predictions_frozen, vect_out = Inference(main_location + "/pb/DNN_" + breath + "_" +
                                                                 quant + "_fact_" + str(lvl) + ".pb", x_test)
                        np.save(main_location + "d-vector-test.npy", vect_out)

                    else:
                        os.system("mkdir " + pb_location + model + "/" + quant)
                        os.system("mkdir " + pb_location + model + "/" + quant + "/fact_" + str(lvl))
                        main_location = pb_location + model + "/" + quant + "/fact_" + str(lvl) + "/"
                        LOG_FILE = open(main_location + 'log.txt', 'a')
                        # lvl_x = min(math.ceil(hid*lvl / 100),hid)
                        build_pb_fact(pb_location + model + "/model.pb", main_location, breath, quant, lvl)

                        # accuracy test
                        predictions_frozen = []
                        predictions_frozen, vect_out = Inference(main_location + "/pb/DNN_" + breath + "_" +
                                                                 quant + "_fact_" + str(lvl) + ".pb", x_test)
                        os.system("zip -r " + main_location + "/pb/compressed.zip " + main_location +
                                  "/pb/DNN_" + breath + "_" + quant + "_fact_" + str(lvl) + ".pb")
                        logEntry(os.system("ls -l " + main_location + "/pb/compressed.zip"))
                        plotConfusionMatrix(list(predictions_frozen), y_test, 100, main_location,
                                            "confusion_matrix.png", "DNN - " + breath + "_" + quant)

                        predictions_frozen, vect_out = Inference(main_location + "/pb/DNN_" + breath + "_" +
                                                                 quant + "_fact_" + str(lvl) + ".pb", x_train)
                        np.save(main_location + "d-vector.npy", vect_out)

                        predictions_frozen, vect_out = Inference(main_location + "/pb/DNN_" + breath + "_" +
                                                                 quant + "_fact_" + str(lvl) + ".pb", x_test)
                        np.save(main_location + "d-vector-test.npy", vect_out)
