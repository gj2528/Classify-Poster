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
print(data.head())

#Next, we load the movie posters.
image_glob = glob.glob(path + "/" + "*.jpg")
print(image_glob)
img_dict = {}

# txtName = "dataset/SampleMoviePoster.txt"
# f = open(txtName, "w")
# for image_single in image_glob:
#     result = image_single +'\n'
#     f.write(result)
# f.close()
# print("writefileDONE")


def get_id(filename):
    index_s = filename.rfind("\\") + 1
    index_f = filename.rfind(".jpg")
    return filename[index_s:index_f]

for fn in image_glob:
    try:
        img_dict[get_id(fn)] = scipy.misc.imread(fn)
    except:
        pass

def show_img(id):
    title = data[data["imdbId"] == int(id)]["Title"].values[0]
    genre = data[data["imdbId"] == int(id)]["Genre"].values[0]
    plt.imshow(img_dict[id])
    plt.title("{} \n {}".format(title, genre))
    plt.show()

#show_img('3772')

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
    #print("dataset:",dataset)
    print("y:",y)
    print("label_dict:",label_dict)
    print("ids:",ids)
    return dataset, y, label_dict, ids, n_classes

#We scale our movie posters to 96×96.我们将电影海报扩展到96×96
SIZE = (150, 101)
dataset, y, label_dict, ids, n_classes = prepare_data(data, img_dict, size=SIZE)

import csv
import codecs
file_name = "dataset/label_all.csv"
def data_write_csv(file_name, datas, ids, flag = False):#file_name为写入CSV文件的路径，datas为要写入数据列表
    out = codecs.open(file_name,'a','utf-8')
    #设定写入模式
    csv_write = csv.writer(out,dialect='excel')
    i = 0
    #写入具体内容
    if (flag):
        datas.insert(0, 'Id')
        csv_write.writerow(datas)
    else:
        for data in datas:
            data = data.tolist()
            data.insert(0, ids[i])
            csv_write.writerow(data)
            i += 1
    out.close()
    print("保存文件成功，处理结束")
# data_write_csv(file_name, label_dict["idx2word"], None, True)
# data_write_csv(file_name, y, ids)

n = 900
n_test = 90

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

    return Z16


# 计算loss
def compute_loss(Z16, Y):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z16, labels=Y))
    return loss


# 定义模型
def model(dataset, y, n_classes, SIZE, learning_rate, num_epochs, minibatch_size, print_cost, isPlot):
    # seed = 3
    n_C0 = 3
    (n_H0, n_W0) = SIZE
    costs = []

    X, Y, keep_prob = create_placeholder(n_H0, n_W0, n_C0, n_classes)

    parameters = init_parameters()

    X_train = dataset[:n]
    Y_train = y[:n]

    # predict

    X_test = dataset[n:n + n_test]
    Y_test = y[n:n + n_test]

    Z16 = forward_propagation(X, parameters, keep_prob)

    cost = compute_loss(Z16, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    num_minibatches = int(n / minibatch_size)  # 获取数据块的数量
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            minibatch_cost = 0

            # seed = seed + 1
            minibatches = random_minibatches(dataset, y, n, BATCH_SIZE)

            # 对每个数据块进行处理
            for minibatch in minibatches:
                # 选择一个数据块
                (minibatch_X, minibatch_Y) = minibatch
                # 最小化这个数据块的成本
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y,keep_prob:0.25})
                #accuracy_score = sess.run(accuracy, feed_dict={X: minibatch_X, Y: minibatch_Y,keep_prob:0.25})
                # 累加数据块的成本值
                minibatch_cost += temp_cost / num_minibatches
            # 是否打印成本
            if print_cost:
                # 每1代打印一次
                if epoch % 1 == 0:
                    print("当前是第 " + str(epoch) + " 代，成本值为：" + str(minibatch_cost) + " 正确率为：" + str(minibatch_accuracy))

                # 记录成本
                if epoch % 1 == 0:
                    costs.append(minibatch_cost)

        # 数据处理完毕，绘制成本曲线
        if isPlot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
        # 保存学习后的参数
        parameters = sess.run(parameters)
        print("参数已经保存到session。")
        saver.save(sess, "model/classify_with_tf/2/save_net.ckpt")

        # 开始预测数据
        ## 计算当前的预测情况
        predict_op = tf.argmax(Z16, 1)

        corrent_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        ##计算准确度
        accuracy = tf.reduce_mean(tf.cast(corrent_prediction, "float"))
        print("corrent_prediction accuracy= " + str(accuracy))

        minibatches = random_minibatches(dataset, y, n, BATCH_SIZE)
        train_acc = 0
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            temp = accuracy.eval({X: minibatch_X, Y: minibatch_Y, keep_prob: 1.0})
            train_acc += temp
        train_accuracy = train_acc/num_minibatches
        test_accuary = accuracy.eval({X: X_test, Y: Y_test, keep_prob:1.0})

        print("训练集准确度：" + str(train_accuracy))
        print("测试集准确度：" + str(test_accuary))

        return parameters

parameters = model(dataset, y, n_classes, SIZE, learning_rate=0.001, num_epochs=5, minibatch_size=4, print_cost=True, isPlot=True)

endtime = datetime.datetime.now()
print("结束时间: ", str(endtime).split('.')[0])
print (u'相差：%s'%(endtime-starttime))

