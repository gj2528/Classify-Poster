import numpy as np
import pandas as pd
import glob
import scipy.misc
import matplotlib
import tensorflow as tf

import matplotlib.pyplot as plt
genres = ['Action', 'Animation', 'Biography','Comedy', 'Crime' , 'Drama' ,  'Horror',  'Romance', 'Sci-Fi']
change_genres = ['Adventure', 'Documentary', 'Thriller']
to_genres = ['Action', 'Biography', 'Horror']
del_genres = [  'Family', 'Fantasy', 'Biography', 'History', 'Mystery', 'War', 'Sport', 'Music', 'Musical', 'Western', 'Short',
 'Film-Noir',  'Talk-Show', 'News','nan' 'Adult', 'Reality-TV', 'Game-Show']
path = 'posters'
data = pd.read_csv("MovieGenre.csv", encoding="ISO-8859-1")
BATCH_SIZE = 128
print(data.head())

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
# import csv
# csvfile=open('dataset/label_all.csv','r')
# csv_reader_lines = csv.reader(csvfile)
# title = []
# labels = []
# flag = True
# for one_line in csv_reader_lines:
#     if flag == True:
#         title.append(one_line)
#     else:
#         label = one_line
#
#         labels.append(one_line)

# for k in img_dict:
#     g = data[data["imdbId"] == int(k)]["Genre"].values[0].split("|")
#     print('g',g)



genre_per_movie = data["Genre"].apply(lambda x: str(x).split("|"))

def cut_label(genre_per_movie):
    lines = []
    i = 0
    for a in genre_per_movie:
        print('i', i)
        i += 1
        l = []
        for b in a:
            #print("g",g)
            if b == "nan":
                pass
            pass
            for c in genres:
                if b == c:
                    g = b
                    l.append(g)
                    continue
                else:
                    for k in range(len(change_genres)):
                        if b == change_genres[k]:
                            g = to_genres[k]
                            if g in l:
                                pass
                            else:
                                l.append(g)
        print('l',l)
        if(len(l)):
            lines.append(l)
        print('lines',lines)
    print('final_lines',lines)
    return lines
# lines = cut_label(genre_per_movie)
#
# def prepare_data(data,lines,img_dict):
#     label_dict = {"word2idx": {}, "idx2word": []}
#     idx = 0
#     y = []
#     ids = []
#     for line in [g for d in lines for g in d]:
#         #print("l",l)
#         if line in label_dict["idx2word"]:
#             pass
#         else:
#             label_dict["idx2word"].append(line)
#             label_dict["word2idx"][line] = idx
#             idx += 1
#     n_classes = len(label_dict["idx2word"])
#     print("identified {} classes".format(n_classes))
#     for k in img_dict:
#         try:
#             g = data[data["imdbId"] == int(k)]["Genre"].values[0].split("|")
#             gg = []
#             #print('gg',gg)
#             for a in g:
#                 if a == "nan":
#                     pass
#                 pass
#                 for c in genres:
#                     if a == c:
#                         s = a
#                         gg.append(s)
#                         continue
#                     else:
#                         for d in range(len(change_genres)):
#                             if a == change_genres[d]:
#                                 b = to_genres[d]
#                                 if b in g:
#                                     pass
#                                 else:
#                                     gg.append(b)
#             print('gg',gg)
#             l = np.sum([np.eye(n_classes, dtype="uint8")[label_dict["word2idx"][s]]
#                         for s in gg], axis=0)  # 列方向上数字相加
#             print('l',l)
#             if(type(l)==list):
#                 y.append(l)
#                 print('y',y)
#                 ids.append(k)
#         except:
#             pass
#     return label_dict,y,ids
# label_dict,y,ids = prepare_data(data,lines,img_dict)
# print('label_dict["idx2word"]',label_dict["idx2word"])
# print('label_dict["word2idx"]',label_dict["word2idx"])
# print('y',y)
# print('len(y)',len(y))
# print('ids',ids)
# print('len(ids)',ids)


#一个简洁的小预处理函数来缩放图像......
def preprocess(img, size=(150, 101)):
    img = scipy.misc.imresize(img, size)
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.
    return img

#a function to generate our data set.
def prepare_data(data,  img_dict, size=(150, 101)):
    print("Generation dataset...")
    dataset = []
    y = []
    ids = []
    label_dict = {"word2idx": {}, "idx2word": []}
    idx = 0
    genre_per_movie = data["Genre"].apply(lambda x: str(x).split("|"))
    genre_per_movie = cut_label(genre_per_movie)
    print('genre_per_movie',genre_per_movie)
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
            gg = []
            #print('gg',gg)
            for a in g:
                if a == "nan":
                    pass
                pass
                for c in genres:
                    if a == c:
                        s = a
                        gg.append(s)
                        continue
                    else:
                        for d in range(len(change_genres)):
                            if a == change_genres[d]:
                                b = to_genres[d]
                                if b in gg:
                                    pass
                                else:
                                    gg.append(b)
            print('gg',gg)
            img = preprocess(img_dict[k], size)
            if img.shape != (150, 101, 3):
                continue
            dataset.append(img)
            if(len(gg)):
                l = np.sum([np.eye(n_classes, dtype="uint8")[label_dict["word2idx"][s]]
                                                            for s in g], axis=0)#列方向上数字相加
                #print('l',l)
                y.append(l)
                ids.append(k)
        except:
            pass
    print("DONE")
    #dataset = np.asarray(dataset)
    y = np.asarray(y)
    print("dataset.type:",type(dataset))
    print("y.type:",type(y))
    #print("dataset:",dataset)
    #print("y:",y)
    print("label_dict:",label_dict)
    print("ids:",ids)
    return dataset, y, label_dict, ids, n_classes

#We scale our movie posters to 96×96.我们将电影海报扩展到96×96
SIZE = (150, 101)
dataset, y, label_dict, ids, n_classes = prepare_data(data, img_dict, size=SIZE)
print('label_dict["idx2word"]',label_dict["idx2word"])
print('label_dict["word2idx"]',label_dict["word2idx"])
print('y',y)
print('ids',ids)

import csv
import codecs
file_name = "data/label.csv"
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
data_write_csv(file_name, label_dict["idx2word"], None, True)
data_write_csv(file_name, y, ids)


