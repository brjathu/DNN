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


quant = "quantization_yes"
breath = "sniff"
LOG_FILE = open("/flush1/raj034/DNN/model/test_cases/" + breath + "/" + quant + "/log.txt", 'a')


def logEntry(TMP_STRING):
    LOG_FILE.write(str(TMP_STRING))
    LOG_FILE.write("\n")
    LOG_FILE.flush()
    print(TMP_STRING)


search_path = "/flush1/raj034/DNN/model/test_cases/" + breath + "/" + "quantization_no" + "/"
tests = os.listdir(search_path)
tests.sort()
tests.remove('log.txt')
print(tests)

for cases in tests:
    pb_file = search_path + str(cases) + "/pb/DNN_" + breath + "_quantization_no_" + str(cases) + ".pb"
    new_path = "/flush1/raj034/DNN/model/test_cases/" + breath + "/" + quant + "/"
    os.system("mkdir " + new_path + "/" + str(cases))
    os.system("mkdir " + new_path + "/" + str(cases) + "/pb")
    qunatized_pb_file = new_path + str(cases) + "/pb/DNN_" + breath + "_" + quant + "_" + str(cases) + ".pb"
    os.system("python converge_weights.py --file " + pb_file + " --save_location " + qunatized_pb_file)
