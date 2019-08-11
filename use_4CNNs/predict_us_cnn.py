import utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

img_path = ("Posters/15532.jpg")

image_id = img_path[8:-4]
image_ids = []
image_ids.append(image_id)
# image_ids = []
# for path in img_path:
#     start = path.rfind("/")+1
#     print('start',start)
#     end = len(path)-4
#     image_ids.append(path[start:end])
print('image_ids:',image_ids)

data = pd.read_csv("MovieGenre.csv", encoding = "ISO-8859-1")
y = []
parsed_movies = []

classes = utils.list_genres(7)
# classes = set(classes)
print('classes:',classes)
print(len(classes))
y = utils.list_movies(classes,image_ids)
print('y:',y)
mlb = MultiLabelBinarizer()
#print('mlb:',mlb)
mlb.fit(y)
#print('mlb:',mlb)
y = mlb.transform(y)
print('y:',y)
print('y.shape:',y.shape)
x = []
x.append(utils.get_image(img_path))
x = np.asarray(x)

print('x.shape:',x.shape)

def predict(X):
    init = tf.global_variables_initializer()

    x,_ = utils.create_placeholders(150, 150, 3, 7)

    parameters = utils.init_weights()

    fc3 = utils.forward_prop(X, parameters)

    p = tf.argmax(fc3, axis = 1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess,tf.train.latest_checkpoint("model/"))
        predict = sess.run(p, feed_dict = {x: X})
        print('predict:', classes[int(predict)])
        # print('predict',predict)     
        # print('predict:',predict) 
        # image = utils.get_image(img_path)
        # plt.imshow(image) 
        # plt.show()

    return predict

predict(x)