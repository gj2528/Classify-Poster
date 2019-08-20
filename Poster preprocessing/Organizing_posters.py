#首先读取对应txt文件中图片的路径，然后根据id将图片放如不同种类的文件夹
import os
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

genres = ['Action','Adventure', 'Animation', 'Biography','Documentary' , 'Comedy', 'Crime' , 'Drama' , 'Family',  'History', 'Horror', 'Thriller','Music','Musical', 'Romance', 'Sci-Fi',  'Sport',  'War']
#genres = ['Sci-Fi']
original_path = "data/"
original_posters_file = ("posters/")


def organizing_posters():
    for genre in genres:
        save_folder = original_path + genre
        isExists = os.path.exists(save_folder)
        if not isExists:
            os.makedirs(save_folder)
            print(save_folder + ' 创建成功')
        else:
            print(save_folder + ' 目录已存在')
        results = []
        try:
            with open(original_path + genre + '.txt', 'r') as f:
                for line in f:
                    str = line.strip('\n')
                    results.append(str)
                print(results)
        except:
            os.removedirs(save_folder)
            print(save_folder + '删除成功')
            pass

        for result in results:
            try:
                image = Image.open(original_posters_file + result + ".jpg")
                image.save(save_folder+'/'+result + ".jpg")
                print("save"+result + ".jpg成功")
            except:
                pass



organizing_posters()
