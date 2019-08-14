import pandas as pd

genres = ['Action','Adventure', 'Animation', 'Biography','Documentary' , 'Comedy', 'Crime' , 'Drama' , 'Family',  'History', 'Horror', 'Thriller','Music','Musical', 'Romance', 'Sci-Fi',  'Sport',  'War']

data = pd.read_csv("dataset/label_all.csv")

for genre in genres:
    f = open("data/" + genre + ".txt", "w")
    #print('data[genre]:',data[genre])
    for i,lb in data[genre].items():
        print(str(i)+':'+str(lb))

        if lb == 1:
            #print('data['+data['Id']+']['+i+']:'+ data['Id'][i])
            s = str(data['Id'][i])+'\n'
            f.write(s)


    f.close()
    print(genre+".txt已完成")

