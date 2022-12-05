# Music Recommendation System

To analyze music currently being played and suggest similar music.
The program suggests music based on user's listening habits and song's streams.

## Dataset Used

The dataset that has been used is Kaggle's Million Songs Dataset. Million Songs Dataset contains of two files: triplet_file and metadata_file. The triplet_file contains user_id, song_id and listen time. The metadata_file contains song_id, title, release, year and artist_name. 

Since the dataset is very large we have added a link to the dataset here. The CSV files used can be downloaded using this link - [song_data.csv](https://drive.google.com/drive/folders/1aFRgsXDsAOeIvpwLNnod67yhyq9gwYe3)

triplet_file has been named song_data_1.csv and metadata_file has been named song_data.csv

## Algorithms

Content Based Filtering (SVD)
```python
def compute_svd(urm, K):
    U, s, Vt = svds(urm, K)

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i,i] = mt.sqrt(s[i])

    U = csc_matrix(U, dtype=np.float32)
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)
    
    return U, S, Vt
```
```python
def compute_estimated_matrix(urm, U, S, Vt, uTest, K, test):
    rightTerm = S*Vt 
    max_recommendation = 250
    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    recomendRatings = np.zeros(shape=(MAX_UID,max_recommendation ), dtype=np.float16)
    for userTest in uTest:
        prod = U[userTest, :]*rightTerm
        estimatedRatings[userTest, :] = prod.todense()
        recomendRatings[userTest, :] = (-estimatedRatings[userTest, :]).argsort()[:max_recommendation]
    return recomendRatings
```
Collaborative Filtering (KNN)
```python
def ItemSimilarityBasedRecommendationEngine(user_id):
    import numpy as np
    uniquesongsofuser=data[data["user_id"]==user_id]["song_id"].unique()
    
    
    uniquesongall=data["song_id"].unique()
    
    m = np.empty((len(uniquesongsofuser),len(uniquesongall)))
    print(m)
    user_song_unique_user_no=[]
    user_song_unique_user=[]

    for song_id in  uniquesongsofuser:
         dict={}
         k=data[data["song_id"]==song_id]
         count=[ i for i in k["user_id"].unique()]
         dict[song_id]=count
         user_song_unique_user.append(dict)
     
         user_song_unique_user_no.append(len(count))
    user_song_unique_all_no=[]
    user_song_unique_all=[]

    time=1
    for song_id in uniquesongall:
         dict={}
         k=data[data["song_id"]==song_id]
         count=[ i for i in k["user_id"].unique()]
         dict[song_id]=count
         user_song_unique_all.append(dict)
         print(time)
         time=time+1
    
     
         user_song_unique_all_no.append(len(count))
    print(m)
    row=0
    column=0
    for i in user_song_unique_user :
        print(row)
        for key,value in i.items():
            user1=value
            song1=key
            for j in user_song_unique_all:
                print(column)
                for key2,value2 in j.items():
                    user2=value2
                    song2=key2
                    if(song1!=song2):
                        m[row,column]=len(np.intersect1d(user1, user2, assume_unique=True))/len(np.union1d(user1,user2))
                    column=column+1
               
        column=0
        row=row+1
    print("total user songs:%d"%len(user_song_unique_user))  
    print("total songs in the training set:%d"%len(user_song_unique_all))
    print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(m))
    k=(m.sum(axis=0)/len(m)).tolist()
    res = sorted(range(len(k)), key = lambda sub: k[sub])
    res.reverse()
    rec=res[:10] 
    alll= uniquesongall.tolist()
    recomm= [alll[index] for index in rec]
    song = pd.read_csv("song_data.csv")
    print("details of songs user listens to")
    list=[]
    print("-----------------------------------------------------------")
    for j in  uniquesongsofuser:
       
        print(song[song['song_id']==j][['title','release','artist_name']])
        list.append(song[song['song_id']==j][['title']])
    print("___________________________________________________________")
    rlist=[]
    print("recommendations *******")
    for j in recomm:
    
        print(song[song['song_id']==j][['title','release','artist_name']])
    rlist.append(song[song['song_id']==j][['title']])
    print("*****************************************************************")
```

## References
[Research paper used as reference](https://cse.iitk.ac.in/users/cs365/2014/_submissions/shefalig/project/report.pdf)

## Authors and acknowledgment
Disha Jain - PES1UG20CS132

Isha Adiraju - PES1UG20CS169

Juhi Bhattacharya - PES1UG20CS182

PES UNIVERSITY, DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING

UE20CS302 - Machine Intelligence