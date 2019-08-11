import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import scipy.misc
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import scipy
import os

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

data = pd.read_csv("MovieGenre.csv", encoding = "ISO-8859-1")

parsed_movies = []

def list_movies(genres, image_ids):
    # for index, row in data.iterrows():
    for image_id in image_ids:
        movies_genres = _parse_movie_row(image_id)
        movies=set(movies_genres).intersection(genres)
        parsed_movies.append(movies)
    result = parsed_movies
    return result

def _parse_movie_row(image_id) :
    genre_str = str(data[data["imdbId"] == int(image_id)]["Genre"].values[0])
    if len(genre_str) > 0:
        genres = genre_str.split('|')
    return genres

def list_genres(number):
    if number == 3:
        return ['Comedy', 'Drama', 'Action']
    if number == 7:
        return list_genres(3) + ['Animation', 'Romance', 'Adventure', 'Horror']
    if number == 14:
        return list_genres(7) + ['Sci-Fi', 'Crime', 'Mystery', 'Thriller', 'War', 'Family', 'Western']

def get_image(image_path):
    image = scipy.misc.imread(image_path)
    image = scipy.misc.imresize(image, (150, 150))
    image = image.astype(np.float32)
    return image

def random_minibatches(x,y,size=64, seed = 0):
	#x = tf.shuffle(x, seed)
	#y = tf.shuffle(y, seed)
	m = x.shape[0]
	nums = m//size
	batches = []
	for k in range(nums):
		mx = x[k*size:(k+1)*size,:,:,:]
		my = y[k*size:(k+1)*size,:]
		batches.append((mx,my))
	mx = x[nums*size:m,:,:,:]
	my = y[nums*size:m,:]
	batches.append((mx,my))
	return batches

def create_placeholders(nh0, nw0, nc0, ny):
    x = tf.placeholder("float", shape = (None, nh0, nw0, nc0))
    y = tf.placeholder("float", shape = (None, ny))
    return x,y

def init_weights():
    w1 = tf.get_variable("w1",[3,3,3,128], initializer = tf.contrib.layers.xavier_initializer_conv2d(seed = 2))
    w2 = tf.get_variable("w2",[3,3,128,64], initializer = tf.contrib.layers.xavier_initializer_conv2d(seed = 2))
    w3 = tf.get_variable("w3",[2,2,64,64], initializer = tf.contrib.layers.xavier_initializer_conv2d(seed = 2))
    w4 = tf.get_variable("w4",[2,2,64,32], initializer = tf.contrib.layers.xavier_initializer_conv2d(seed = 2))
    params = {"w1":w1,"w2":w2,"w3":w3,"w4":w4}
    return params

def forward_prop(x, params):
    w1 = params["w1"]
    w2 = params["w2"]
    w3 = params["w3"]
    w4 = params["w4"]
    z1 = tf.nn.conv2d(x,w1,[1,1,1,1],padding = 'SAME')
    a1 = tf.nn.relu(z1)
    dp1 = tf.nn.dropout(a1, 0.7)
        
    z2 = tf.nn.conv2d(dp1,w2,[1,1,1,1], padding = 'SAME')
    a2 = tf.nn.relu(z2)
    p2 = tf.nn.max_pool(a2, ksize = [1,3,3,1], strides = [1,1,1,1], padding = 'SAME')
    dp2 = tf.nn.dropout(p2, 0.7)
    
    z3 = tf.nn.conv2d(dp2, w3, [1,1,1,1],padding = 'SAME')
    a3 = tf.nn.relu(z3)
    p3 = tf.nn.max_pool(a3, ksize = [1,2,2,1], strides = [1,1,1,1], padding = 'SAME')
    dp3 = tf.nn.dropout(p3, 0.8)
    
    z4 = tf.nn.conv2d(dp3, w4, [1,1,1,1],padding = 'SAME')
    a4 = tf.nn.relu(z4)
    dp4 = tf.nn.dropout(a4, 0.8)
    #print('dp4.shape:',dp4.shape)
    
    fc1 = tf.contrib.layers.flatten(dp4)
    fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs = 46, activation_fn = tf.nn.relu)
    fc3 = tf.contrib.layers.fully_connected(fc2, num_outputs = 7, activation_fn = None)
    return fc3