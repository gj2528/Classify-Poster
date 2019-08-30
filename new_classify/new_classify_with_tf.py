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

path = 'data/all_organizing_posters'
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

genres, trains, labels = get_data(label_path)
# print('genres',genres)
# print('trains',trains)
# print('labels',labels)

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

for fn in image_glob:
    try:
        img_dict[get_id(fn)] = scipy.misc.imread(fn)
    except:
        pass
print('img_dict',img_dict)

def show_img(id,genres,trains,labels):
    title = data[data["imdbId"] == int(id)]["Title"].values[0]
    genre = get_genre(id,genres,trains,labels)
    plt.imshow(img_dict[id])
    plt.title("{} \n {}".format(title, genre))
    plt.show()

#show_img('2461',genres,trains,labels)

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

# def prepare_data(img_dict, genres, trains, labels, size=(150, 101)):
#     print("Generation dataset...")
#     dataset = []
#     y = []
#     ids = []
#     n_classes = len(genres)
#     for k in img_dict:
#         try:
#             img = preprocess(img_dict[k], size)
#             if img.shape != (150, 101, 3):
#                 continue
#             l = get_label(k,trains, labels)
#             y.append(l)
#             dataset.append(img)
#             ids.append(k)
#         except:
#             pass
#     print("DONE")
#     dataset = np.asarray(dataset)
#     y = np.asarray(y)
#     return dataset, y, ids, n_classes


#We scale our movie posters to 96×96.我们将电影海报扩展到96×96
SIZE = (150, 101)
dataset, y, ids, n_classes = prepare_data(img_dict, genres, trains, labels, size=SIZE)
print('dataset.shape',dataset.shape)
print('y.shape',y.shape)
print('ids',ids)
print('n_classes',n_classes)

n = 6000

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



    X_train = dataset[:n]
    Y_train = y[:n]

    # predict
    n_test = 530
    X_test = dataset[n:n + n_test]
    Y_test = y[n:n + n_test]

    Z16 = forward_propagation(X, keep_prob)

    cost = compute_loss(Z16, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            minibatch_cost = 0
            num_minibatches = int(n / minibatch_size)  # 获取数据块的数量
            # seed = seed + 1
            minibatches = random_minibatches(dataset, y, n, BATCH_SIZE)

            # 对每个数据块进行处理
            for minibatch in minibatches:
                # 选择一个数据块
                (minibatch_X, minibatch_Y) = minibatch
                # 最小化这个数据块的成本
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y,keep_prob:0.25})

                # 累加数据块的成本值
                minibatch_cost += temp_cost / num_minibatches

            # 是否打印成本
            if print_cost:
                # 每1代打印一次
                if epoch % 1 == 0:
                    print("当前是第 " + str(epoch) + " 代，成本值为：" + str(minibatch_cost))

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

        saver.save(sess, "model/classify_with_tf/5/save_net.ckpt")
        print("参数已经保存到session。")

        # 开始预测数据
        ## 计算当前的预测情况
        predict_op = tf.argmax(Z16, 1)

        corrent_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        ##计算准确度
        accuracy = tf.reduce_mean(tf.cast(corrent_prediction, "float"))
        print("corrent_prediction accuracy= " + str(accuracy))

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train, keep_prob:1.0})
        test_accuary = accuracy.eval({X: X_test, Y: Y_test, keep_prob:1.0})

        print("训练集准确度：" + str(train_accuracy))
        print("测试集准确度：" + str(test_accuary))



model(dataset, y, n_classes, SIZE, learning_rate=0.001, num_epochs=5, minibatch_size=4, print_cost=True, isPlot=True)

endtime = datetime.datetime.now()
print("结束时间: ", str(endtime).split('.')[0])
print (u'相差：%s'%(endtime-starttime))

