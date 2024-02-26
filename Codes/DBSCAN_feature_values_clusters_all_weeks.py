import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.decomposition import PCA
from tqdm import tqdm

path = ''
savedata = ''

dropcols = ['spy','amd','tsla','mu','aapl','amzn','msft','snap','nvda','spce','fb','dis','bynd','nflx','jnug','ge','rad','sq','atvi','uso','twtr','amc','bb','nok','pltr','gme']

df = pd.DataFrame()

for i in range(0,178):
    
    features = pd.read_csv(path + 'week_{}.csv'.format(i), index_col = 0)
    features.drop(columns = dropcols, inplace = True)
    features = features.sample(10_000)
    features.reset_index(drop = True, inplace = True)
    
    M = features.drop(columns = ['author'])
    
    pca = PCA(n_components = 10)
    X = pca.fit(M).transform(M)
        
    db = DBSCAN(eps=1, min_samples=15).fit(X)
    features['labels'] = db.labels_
    sizes = features.labels.value_counts()
    sizes.drop(-1, inplace = True)
    sizes = sizes[0:3]
    
    n1 = features.labels == sizes.index[0]   #commenters have label 0
    features.labels[n1] = 0
    if(len(sizes) > 1):
        n2 = features.labels == sizes.index[1]   #active have label 1
        features.labels[n2] = 1
    if(len(sizes) > 2):
        n3 = features.labels == sizes.index[2]   #posters have label 2
        features.labels[n3] = 2
    
    m1 = features.labels == 0   #commenters have label 0
    m2 = features.labels == 1   #active have label 1
    m3 = features.labels == 2   #posters have label 2
    
    mtot = (m1 | m2) | m3
    features = features[mtot]
    
    df = pd.concat([df,features], ignore_index=True)
    
df.to_csv(savedata + 'features_in_clusters_all_weeks.csv')
