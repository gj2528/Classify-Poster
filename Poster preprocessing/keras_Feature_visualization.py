import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import pandas as pd
import glob
import scipy.misc
import numpy as np
import datetime
starttime = datetime.datetime.now()
print("当前时间: ", str(starttime).split('.')[0])

#genres = ['Action', 'Animation', 'Biography','Comedy', 'Crime' , 'Drama' ,  'Horror',  'Romance', 'Sci-Fi']
genres = ['Action']
path = 'data/'
data = pd.read_csv("MovieGenre.csv", encoding="ISO-8859-1")
BATCH_SIZE = 700
#print(data.head())

#Next, we load the movie posters.
#txt_path = glob.glob(path + "/" + "*.txt")
#print(txt_path)

img_dict = {}

results = []
def get_id(path):
    with open(path, 'r') as f:
        for line in f:
            str = line.strip('\n')
            results.append(str)
        print(results)
    return results

#一个简洁的小预处理函数来缩放图像......
def preprocess(img, size=(150, 101)):
    img = scipy.misc.imresize(img, size)
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.
    return img



def show_img(id):
    title = data[data["imdbId"] == int(id)]["Title"].values[0]
    genre = data[data["imdbId"] == int(id)]["Genre"].values[0]
    plt.imshow(img_dict[id])
    plt.title("{} \n {}".format(title, genre))
    plt.show()


#show_img('3772')

def prepare_data():
    dataset_dict = {}.fromkeys(genres, [])
    print("dataset_dict_orign:", dataset_dict)
    dataset_num_dict = {}.fromkeys(genres, 0)
    print("dataset_num_dict_orign:", dataset_num_dict)
    for genre in genres:
        txt_path = path + genre + ".txt"
        image_path = path + genre + "/"
        ids = get_id(txt_path)
        for id in ids:
            try:
                img_dict[id] = scipy.misc.imread(image_path + id + ".jpg")
                img = preprocess(img_dict[id])
                if img.shape != (150, 101, 3):
                    continue
                dataset_dict[genre].append(img)
                #dataset_dict = np.asarray(dataset_dict)
                dataset_num_dict[genre] += 1
            except:
                pass
        print('dataset_dict [genre][0]的type', type(dataset_dict[genre][0]))
        print('dataset_dict [genre]的type', type(dataset_dict[genre]))
        print('dataset_num_dict [genre]的type', type(dataset_num_dict[genre]))
    #print("dataset_dict_last:", dataset_dict)
    print('dataset_dict设定后的type', type(dataset_dict))
    print('dataset_num_dict',dataset_num_dict)
    return dataset_dict, dataset_num_dict

dataset_dict, dataset_num_dict = prepare_data()

model = load_model('model/classify_model/classify_model.h5') #replaced by your model name
#layer_1 = K.function([model.layers[0].input], [model.layers[1].output])#第一个 model.layers[0],不修改,表示输入数据；第二个model.layers[you wanted],修改为你需要输出的层数的编号
layer_5 = K.function([model.layers[0].input], [model.layers[5].output])#第7个 model.layers[0],不修改,表示输入数据；第二个model.layers[you wanted],修改为你需要输出的层数的编号
#layer_8 = K.function([model.layers[0].input], [model.layers[8].output])#第7个 model.layers[0],不修改,表示输入数据；第二个model.layers[you wanted],修改为你需要输出的层数的编号


for genre in genres:
    for p in range(dataset_num_dict[genre]):
        input_image = dataset_dict[genre][p]
        input_image = input_image.reshape(1,150,101,3)
        print("input_image",input_image)
        print("input_image的type",type(input_image))
        print("input_image的shape",input_image.shape)
        # f1 = layer_1([input_image])[0]  # 只修改input_image
        # # 第一层卷积后的特征图展示，输出是（1,146,97,32），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
        # for _ in range(32):
        #     show_img = f1[:, :, :, _]
        #     show_img.shape = [146, 97]
        #     plt.subplot(4, 8, _ + 1)
        #     plt.imshow(show_img, cmap='gray')
        #     plt.axis('off')
        # plt.show()
        f7 = layer_5([input_image])[0]  # 只修改input_image
        # 第一层卷积后的特征图展示，输出是（1,34,22,64），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
        for _ in range(4):
            show_img = f7[:, :, :, _]
            show_img.shape = [69, 44]
            plt.subplot(2, 2, _ + 1)
            plt.imshow(show_img, cmap='gray')
            plt.axis('off')
        plt.show()
        # f1 = layer_8([input_image])[0]
        # print("f1",f1)


endtime = datetime.datetime.now()
print("结束时间: ", str(endtime).split('.')[0])
print (u'相差：%s'%(endtime-starttime))









