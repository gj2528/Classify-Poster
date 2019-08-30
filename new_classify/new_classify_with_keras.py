import numpy as np
import pandas as pd
import glob
import scipy.misc
import matplotlib
import os
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

#import the movie metadata.
import datetime
starttime = datetime.datetime.now()
print("当前时间: ", str(starttime).split('.')[0])

path = 'data/all_organizing_posters'
data = pd.read_csv("MovieGenre.csv", encoding="ISO-8859-1")
BATCH_SIZE = 700
print(data.head())

#Next, we load the movie posters.
image_glob = glob.glob(path + "/" + "*.jpg")
#print(image_glob)
img_dict = {}


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

#show_img('3772',genres,trains,labels)

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

#We scale our movie posters to 96×96.我们将电影海报扩展到96×96
SIZE = (150, 101)
dataset, y, ids, n_classes = prepare_data(img_dict, genres, trains, labels, size=SIZE)

#build the model

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=(SIZE[0], SIZE[1], 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='sigmoid'))



model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()

# from keras.utils import plot_model
# plot_model(model, to_file='model/classify_model/model.png',show_shapes=True)
# print("model.png保存成功")

n = 6000
# model.fit(np.array(dataset[: n]), np.array(y[: n]), batch_size=16, epochs=5,
#           verbose=1, validation_split=0.1)
batches = int(n / BATCH_SIZE)
filepath = "weight/new_classify_with_keras/classify_weights-improvement-{epoch:02d}-{acc:.2f}.hdf5"
# 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]

if os.path.exists(filepath):
    model.load_weights(filepath)
    # 若成功加载前面保存的参数，输出下列信息
    print("checkpoint_loaded")
for p in range(batches):
    print('round:', p)
    traindata = np.array(dataset[p * BATCH_SIZE:(p + 1) * BATCH_SIZE])
    print('image.shape: ', traindata[0].shape)
    trainlabel = np.array(y[p * BATCH_SIZE:(p + 1) * BATCH_SIZE])
    model.fit(x=traindata,
              y=trainlabel,
              epochs=5,
              batch_size=100,
              callbacks=callbacks_list,
              verbose=1,
              validation_split=0.1
              # validation_data=(validation_dataset, labelsValidation)
              )
model.save('model/new_classify_model.h5')

#predict
n_test = 530
X_test = dataset[n:n + n_test]
y_test = y[n:n + n_test]

pred = model.predict(np.array(X_test))
print('pred:',pred)
print('pred.shape:',pred.shape)

def show_example(idx):
    N_true = int(np.sum(y_test[idx]))
    print('N_true',N_true)
    show_img(ids[n + idx],genres,trains,labels)
    image = image_glob[n+idx]
    id = get_id(image)
    print("id",id)
    print("True: " + str(get_genre(id, genres, trains, labels)))
    print("Prediction: {}".format("|".join(["{} ({:.3})".format(genres[s],
                                                                pred[idx][s])
                                            for s in pred[idx].argsort()[-N_true:][::-1]])))

show_example(3)

show_example(48)

show_example(68)

endtime = datetime.datetime.now()
print("结束时间: ", str(endtime).split('.')[0])
print (u'相差：%s'%(endtime-starttime))
