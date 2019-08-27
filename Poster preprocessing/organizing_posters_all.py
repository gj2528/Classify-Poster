import os
from PIL import Image

label_path = 'data/label1.csv'
original_posters_file = 'posters/'
def get_data(label_path):
    with open(label_path, 'r')as f:
        trains = []
        lines = f.readlines()[1:]
        #print('lines', lines)
        for line in lines:
            line = line[:-1]
            label = line.split(',')

            trains.append(label[0])  # 读取第一列的数据
        f.close()
    return trains
trains = get_data(label_path)

save_folder = "data/all_organizing_posters"
isExists = os.path.exists(save_folder)
if not isExists:
    os.makedirs(save_folder)
    print(save_folder + ' 创建成功')
else:
    print(save_folder + ' 目录已存在')

def organizing_posters():
    for train in trains:
        try:
            image = Image.open(original_posters_file + train + ".jpg")
            image.save(save_folder+'/'+train + ".jpg")
            print("save"+train + ".jpg成功")
        except:
            pass
organizing_posters()