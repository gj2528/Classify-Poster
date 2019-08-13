import numpy as np
import pandas as pd
import glob
import scipy.misc
import matplotlib
import tensorflow as tf

import matplotlib.pyplot as plt

#import the movie metadata.



def get_id(filename):
    index_s = filename.rfind("\\") + 1
    index_f = filename.rfind(".jpg")
    return filename[index_s:index_f]



#一个简洁的小预处理函数来缩放图像......
def preprocess(img, size=(150, 101)):
    img = scipy.misc.imresize(img, size)
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.
    return img

#a function to generate our data set.
def prepare_data(data, img_dict, size=(150, 101)):
    print("Generation dataset...")
    dataset = []
    y = []
    ids = []
    label_dict = {"word2idx": {}, "idx2word": []}
    idx = 0
    genre_per_movie = data["Genre"].apply(lambda x: str(x).split("|"))
    for l in [g for d in genre_per_movie for g in d]:
        #print("l",l)
        if l in label_dict["idx2word"]:
            pass
        else:
            label_dict["idx2word"].append(l)
            label_dict["word2idx"][l] = idx
            idx += 1
    n_classes = len(label_dict["idx2word"])
    print("identified {} classes".format(n_classes))
    n_samples = len(img_dict)
    print("got {} samples".format(n_samples))
    for k in img_dict:
        try:
            g = data[data["imdbId"] == int(k)]["Genre"].values[0].split("|")
            img = preprocess(img_dict[k], size)
            if img.shape != (150, 101, 3):
                continue
            l = np.sum([np.eye(n_classes, dtype="uint8")[label_dict["word2idx"][s]]
                                                        for s in g], axis=0)#列方向上数字相加
            y.append(l)
            dataset.append(img)
            ids.append(k)
        except:
            pass
    print("DONE")
    dataset = np.asarray(dataset)
    y = np.asarray(y)
    #print("dataset:",dataset)
    #print("y:",y)
    print("label_dict:",label_dict)
    print("ids:",ids)
    return dataset, y, label_dict, ids, n_classes



def random_minibatches(dataset, y, n, BATCH_SIZE):
    nums = int(n / BATCH_SIZE)
    batches = []
    for p in range(nums):
        traindata = np.asarray(dataset[p * BATCH_SIZE:(p + 1) * BATCH_SIZE])
        trainlabel = np.asarray(y[p * BATCH_SIZE:(p + 1) * BATCH_SIZE])
        batches.append((traindata, trainlabel))
    traindata = np.asarray(dataset[nums * BATCH_SIZE:n])
    trainlabel = np.asarray(y[nums * BATCH_SIZE:n])
    batches.append((traindata, trainlabel))
    return batches


# 创建占位符
def create_placeholder(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name="X")
    Y = tf.placeholder(tf.float32, [None, n_y], name="Y")
    keep_prob = tf.placeholder(tf.float32)
    return X, Y, keep_prob


# 初始化参数
def init_parameters():
    tf.set_random_seed(1)  # 指定随机种子
    W1 = tf.get_variable("W1", [3, 3, 3, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b1 = tf.get_variable("b1", [32, ], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [3, 3, 32, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b2 = tf.get_variable("b2", [32, ], initializer=tf.zeros_initializer())

    W3 = tf.get_variable("W3", [3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b3 = tf.get_variable("b3", [64, ], initializer=tf.zeros_initializer())
    W4 = tf.get_variable("W4", [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b4 = tf.get_variable("b4", [64, ], initializer=tf.zeros_initializer())

    # W5 = tf.get_variable("W5", [3, 3, 32, 48], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    # b5 = tf.get_variable("b5", [48, ], initializer=tf.zeros_initializer())
    # W6 = tf.get_variable("W6", [3, 3, 48, 48], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    # b6 = tf.get_variable("b6", [48, ], initializer=tf.zeros_initializer())
    # W7 = tf.get_variable("W7", [3, 3, 48, 48], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    # b7 = tf.get_variable("b7", [48, ], initializer=tf.zeros_initializer())

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4}

    return parameters


# 前向传播
def forward_propagation(X, parameters, keep_prob):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']

    Z1 = tf.nn.bias_add(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="VALID"), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.nn.bias_add(tf.nn.conv2d(A1, W2, strides=[1, 1, 1, 1], padding="VALID"), b2)
    A2 = tf.nn.relu(Z2)
    P1 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    D1 = tf.nn.dropout(P1,keep_prob=keep_prob)
    print("p1"+str(P1.shape))

    Z3 = tf.nn.bias_add(tf.nn.conv2d(D1, W3, strides=[1, 1, 1, 1], padding="VALID"), b3)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.nn.bias_add(tf.nn.conv2d(A3, W4, strides=[1, 1, 1, 1], padding="VALID"), b4)
    A4 = tf.nn.relu(Z4)
    P2 = tf.nn.max_pool(A4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    D2 = tf.nn.dropout(P2, keep_prob=keep_prob)
    print("p2"+str(P2.shape))

    # f1
    F1 = tf.contrib.layers.flatten(D2)
    Z14 = tf.contrib.layers.fully_connected(F1, 128, activation_fn=tf.nn.relu)  # None)#tf.nn.relu) tf.nn.sigmoid
    D3 = tf.nn.dropout(Z14, keep_prob=2*keep_prob)
    # Z15 = tf.contrib.layers.fully_connected(D1, 32, activation_fn=tf.nn.relu)
    # D2 = tf.nn.dropout(Z15, 0.5)
    Z16 = tf.contrib.layers.fully_connected(D3, 29, activation_fn=None)
    print("z16" + str(Z16.shape))

    return Z16

