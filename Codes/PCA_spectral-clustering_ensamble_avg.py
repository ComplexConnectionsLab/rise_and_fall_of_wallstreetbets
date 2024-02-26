import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import networkx as nx
import numpy.linalg
import matplotlib
matplotlib.style.use("seaborn-talk")
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

path = ''
savedata = ''

dropcols = ['spy','amd','tsla','mu','aapl','amzn','msft','snap','nvda','spce','fb','dis','bynd','nflx','jnug','ge','rad','sq','atvi','uso','twtr','amc','bb','nok','pltr','gme']

w = [23, 54, 84, 115, 145, 176]

cols = list(np.arange(0,100))
cols = list(map(str, cols))

df = pd.DataFrame(0, index=np.arange(len(w)), columns=cols)
nf = pd.DataFrame(0, index=np.arange(len(w)), columns=cols)
cf = pd.DataFrame(0, index=np.arange(len(w)), columns=cols)

for j in range(0,100):
    
    clusters = []
    clus_numb = []
    noise = []
     
    for i in w:
        
        features = pd.read_csv(path + 'week_{}.csv'.format(i), index_col = 0)
        features.drop(columns = dropcols, inplace = True)
    
        features = features.sample(10_000) #prende ogni volta delle sample diverse
        features.reset_index(drop = True, inplace = True)
                
        M = features.drop(columns = ['author'])
        
        pca = PCA(n_components = 10)
        X = pca.fit(M).transform(M) 
        
        dis = pairwise_distances(X) 
        s = np.exp(-dis**2)
        np.fill_diagonal(s, 0)
        
        s[s < 0.5] = 0
        
        g = nx.from_numpy_matrix(s)
        
        isolates = list(nx.isolates(g))
        
        conn = nx.number_connected_components(g)
            
        l = [len(c) for c in sorted(nx.connected_components(g), key=len, reverse=True)]
        
        #add to noise ponts from connected components with less than 5 nodes
        cc = [c for c in sorted(nx.connected_components(g), key=len, reverse=True)]
                
        for comp in cc:
            if len(comp) < 5:
                isolates.extend(list(comp))
    
        isolates = list(set(isolates))
        
        g.remove_nodes_from(isolates)
        
        noise.append(len(isolates))
        
        f = [k for k in l if k >= 5]
        
        clusters.append(f)
        clus_numb.append(len(f))
    
    
    df['{}'.format(j)] = clus_numb
    nf['{}'.format(j)] = noise
    cf['{}'.format(j)] = clusters   
    
    
df.to_csv(savedata + 'cluster_numbers.csv') 
nf.to_csv(savedata + 'noise.csv')
cf.to_csv(savedata + 'cluster_sizes.csv')   
