import numpy as np
import pandas as pd
import glob
import scipy.misc
import matplotlib
import tensorflow as tf

import matplotlib.pyplot as plt

#import the movie metadata.
import datetime
starttime = datetime.datetime.now()
print("当前时间: ", str(starttime).split('.')[0])

path = 'posters'
data = pd.read_csv("MovieGenre.csv", encoding="ISO-8859-1")
BATCH_SIZE = 128
#print(data.head())

label_path = 'data/label1.csv'

#读取csv文件获取label
def get_data(label_path):
    with open(label_path, 'r')as f:
        genres = []
        trains = []
        labels = []
        lines = f.readlines()[1:]
        #print('lines', lines)
        for line in lines:
            line = line[:-1]
            label = line.split(',')
            labels.append(label[1:])  # 读取第二列以后的数据

            trains.append(label[0])  # 读取第一列的数据
        f.close()
    with open(label_path, 'r')as f:
        first_line = f.readlines()[0]
        first_line = first_line[:-1]
        title = first_line.split(',')
        # print('title',title)
        # print(type(title[0]))
        # print('title.type',type(title))
        for t in title:
            genres.append(t)
            #print('genres',genres)
        genres.remove('Id')
        #print('genres', genres)
        f.close()
    return genres,trains,labels

def get_label(id,trains,labels):
    i = 0
    for train in trains:
        # print('i', i)
        # print('train', train)
        ls = []
        if train == id:
            label = labels[i]
            for l in label:
                l = int(l)
                ls.append(l)
            break
        else:
            i += 1
    return ls
# label = get_label('2461',trains, labels)
# print('label',label)

def get_genre(id,genres,trains,labels):
    #print('id',id)
    dataset_genres = []
    i = 0
    for train in trains:
        # print('i',i)
        # print('train',train)
        if train == id:
            j = 0
            for label in labels[i]:
                # print('j', j)
                # print('label', label)
                if label == '1':
                    dataset_genres.append(genres[j])
                    # print('dataset_genres', dataset_genres)
                    j += 1
                else:
                    j += 1
            break
        else:
            i += 1
    return dataset_genres

#Next, we load the movie posters.
image_glob = glob.glob(path + "/" + "*.jpg")
print(image_glob)
img_dict = {}

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
def prepare_data(img_dict, genres, trains, labels, size=(150, 101)):
    print("Generation dataset...")
    dataset = []
    y = []
    ids = []
    n_classes = len(genres)
    for train in trains:
        try:
            img = preprocess(img_dict[train], size)
            if img.shape != (150, 101, 3):
                continue
            l = get_label(train,trains, labels)
            y.append(l)
            dataset.append(img)
            ids.append(train)
        except:
            pass
    print("DONE")
    dataset = np.asarray(dataset)
    y = np.asarray(y)
    return dataset, y, ids, n_classes


def random_minibatches(dataset, y, n, BATCH_SIZE):
    nums = int(n / BATCH_SIZE)
    batches = []
    for p in range(nums):
        traindata = dataset[p * BATCH_SIZE:(p + 1) * BATCH_SIZE]
        trainlabel = y[p * BATCH_SIZE:(p + 1) * BATCH_SIZE]
        batches.append((traindata, trainlabel))
    traindata = dataset[nums * BATCH_SIZE:n]
    trainlabel = y[nums * BATCH_SIZE:n]
    batches.append((traindata, trainlabel))
    return batches


# 创建占位符
def create_placeholder(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name="X")
    Y = tf.placeholder(tf.float32, [None, n_y], name="Y")
    keep_prob = tf.placeholder(dtype=tf.float32)
    return X, Y, keep_prob

# 前向传播
def forward_propagation(X, keep_prob):
    Z1 = tf.contrib.layers.conv2d(inputs=X, num_outputs=32, kernel_size=[3, 3], stride=[1, 1], padding='VALID',
                                  activation_fn=tf.nn.relu)
    Z2 = tf.contrib.layers.conv2d(inputs=Z1, num_outputs=32, kernel_size=[3, 3], stride=[1, 1], padding='VALID',
                                  activation_fn=tf.nn.relu)
    P1 = tf.contrib.layers.max_pool2d(inputs=Z2, kernel_size=[2, 2], stride=[2, 2], padding='VALID')
    D1 = tf.nn.dropout(P1,keep_prob=keep_prob)
    print("p1"+str(P1.shape))

    Z3 = tf.contrib.layers.conv2d(inputs=D1, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='VALID',
                                  activation_fn=tf.nn.relu)
    Z4 = tf.contrib.layers.conv2d(inputs=Z3, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='VALID',
                                  activation_fn=tf.nn.relu)
    P2 = tf.contrib.layers.max_pool2d(inputs=Z4, kernel_size=[2, 2], stride=[2, 2], padding='VALID')
    D2 = tf.nn.dropout(P2, keep_prob=keep_prob)
    print("p2"+str(P2.shape))

    # f1
    F1 = tf.contrib.layers.flatten(D2)
    Z14 = tf.contrib.layers.fully_connected(F1, 128, activation_fn=tf.nn.relu)  # None)#tf.nn.relu) tf.nn.sigmoid
    D3 = tf.nn.dropout(Z14, keep_prob=2*keep_prob)
    Z16 = tf.contrib.layers.fully_connected(D3, 8, activation_fn=None)

    return Z16