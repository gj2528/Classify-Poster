import classify_with_tf_utils as utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.misc
import glob
from scipy import ndimage
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

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

for fn in image_glob:
    try:
        img_dict[utils.get_id(fn)] = scipy.misc.imread(fn)
    except:
        pass

def show_img(id):
    title = data[data["imdbId"] == int(id)]["Title"].values[0]
    genre = data[data["imdbId"] == int(id)]["Genre"].values[0]
    plt.imshow(img_dict[id])
    plt.title("{} \n {}".format(title, genre))
    plt.show()

SIZE = (150, 101)
dataset, y, label_dict, ids, n_classes = utils.prepare_data(data, img_dict, size=SIZE)
print('dataset',dataset)
print('y', y)
n = 35000

# predict
n_test = 3000
X_test = dataset[n:n + n_test]
y_test = y[n:n + n_test]

def show_example(idx,pred):
    N_true = int(np.sum(y_test[idx]))
    print('N_true', N_true)
    show_img(ids[n + idx])
    image = image_glob[n+idx]
    id = utils.get_id(image)
    print("id",id)
    print("Prediction: {}".format("|".join(["{} ({:.3})".format(label_dict["idx2word"][s],
                                                                pred[0][s])
                                            for s in pred[0].argsort()[-N_true:][::-1]])))

def predict():
    init = tf.global_variables_initializer()

    x,y,keep_prob = utils.create_placeholder(150, 101, 3, 29)
    #Z = tf.placeholder(tf.float32, [-1,29], name="Z")
    parameters = utils.init_parameters()

    z16 = utils.forward_propagation(x, parameters,keep_prob)
    print("z16" + str(z16.shape))
    print(z16)

    p = tf.cast(z16, "float")
    print("p" + str(p.shape))
    print(p)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess,tf.train.latest_checkpoint("model/classify_with_tf/1"))
        m = 5
        i = 0
        while(m>0):

            X = dataset[n+i]
            X = X.reshape(1, 150, 101, 3)

            predict = sess.run(p, feed_dict = {x: X,keep_prob:1.0})
            print('predict:',predict)

            show_example(i,predict)
            i += 1
            m -= 1



predict()


endtime = datetime.datetime.now()
print("结束时间: ", str(endtime).split('.')[0])
print (u'相差：%s'%(endtime-starttime))
