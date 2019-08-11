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

path = 'SampleMoviePosters'
data = pd.read_csv("MovieGenre.csv", encoding="ISO-8859-1")
BATCH_SIZE = 700
print(data.head())

#Next, we load the movie posters.
image_glob = glob.glob(path + "/" + "*.jpg")
#print(image_glob)
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

def show_img(id):
    title = data[data["imdbId"] == int(id)]["Title"].values[0]
    genre = data[data["imdbId"] == int(id)]["Genre"].values[0]
    plt.imshow(img_dict[id])
    plt.title("{} \n {}".format(title, genre))
    plt.show()

show_img('3772')

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
    #print("y:",y)
    print("label_dict:",label_dict)
    print("ids:",ids)
    return dataset, y, label_dict, ids

#We scale our movie posters to 96×96.我们将电影海报扩展到96×96
SIZE = (150, 101)
dataset, y, label_dict, ids =  prepare_data(data, img_dict, size=SIZE)

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
model.add(Dense(29, activation='sigmoid'))



model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()

n = 900
# model.fit(np.array(dataset[: n]), np.array(y[: n]), batch_size=16, epochs=5,
#           verbose=1, validation_split=0.1)
batches = int(n / BATCH_SIZE)
filepath = "weight/classify/classify_weights-improvement-{epoch:02d}-{acc:.2f}.hdf5"
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
              epochs=1,
              batch_size=100,
              callbacks=callbacks_list,
              verbose=1,
              validation_split=0.1
              # validation_data=(validation_dataset, labelsValidation)
              )
model.save('model/classify_model_sample.h5')

#predict
n_test = 90
X_test = dataset[n:n + n_test]
y_test = y[n:n + n_test]

pred = model.predict(np.array(X_test))
print('pred:',pred)
print('pred.shape:',pred.shape)

def show_example(idx):
    N_true = int(np.sum(y_test[idx]))
    print('N_true',N_true)
    show_img(ids[n + idx])
    image = image_glob[n+idx]
    id = get_id(image)
    print("id",id)
    print("Prediction: {}".format("|".join(["{} ({:.3})".format(label_dict["idx2word"][s],
                                                                pred[idx][s])
                                            for s in pred[idx].argsort()[-N_true:][::-1]])))

show_example(3)

show_example(48)

show_example(68)

endtime = datetime.datetime.now()
print("结束时间: ", str(endtime).split('.')[0])
print (u'相差：%s'%(endtime-starttime))
