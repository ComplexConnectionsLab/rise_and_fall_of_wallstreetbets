import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.decomposition import PCA
from tqdm import tqdm

path = ''
savedata = ''

dropcols = ['spy','amd','tsla','mu','aapl','amzn','msft','snap','nvda','spce','fb','dis','bynd','nflx','jnug','ge','rad','sq','atvi','uso','twtr','amc','bb','nok','pltr','gme']

cols = list(np.arange(0,100))
cols = list(map(str, cols))

df = pd.DataFrame(0, index=np.arange(178), columns=cols)
nf = pd.DataFrame(0, index=np.arange(178), columns=cols)
cf = pd.DataFrame(0, index=np.arange(178), columns=cols)

for j in tqdm(range(0, 100)):

    noisepoints = []
    clusters = []
    sizes = []
    
    for i in range(0,178):
        
        features = pd.read_csv(path + 'week_{}.csv'.format(i), index_col = 0)
        features.drop(columns = dropcols, inplace = True)
        features = features.sample(10_000)
        features.reset_index(drop = True, inplace = True)
     
        
        M = features.drop(columns = ['author'])
        
        pca = PCA(n_components = 10)
        X = pca.fit(M).transform(M)
            
        db = DBSCAN(eps=1, min_samples=10).fit(X)
        
        l = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        n_noise_ = list(db.labels_).count(-1)
         
        noisepoints.append(n_noise_)
        clusters.append(l)
        
        count = pd.Series(db.labels_).value_counts()
        count = count[count.index != -1]
        sizes.append(list(count.values))
        
        
    df['{}'.format(j)] = clusters
    nf['{}'.format(j)] = noisepoints
    cf['{}'.format(j)] = sizes


df.to_csv(savedata + 'cluster_numbers.csv')
nf.to_csv(savedata + 'noisepoints.csv')
cf.to_csv(savedata + 'cluster_sizes.csv')
