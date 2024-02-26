import pandas as pd
import numpy as np
import networkx as nx
import numpy.linalg
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

path = ''
savedata = ''

tickers = ['spy',
       'amd', 'tsla', 'mu', 'aapl', 'amzn', 'msft', 'snap', 'nvda', 'spce',
       'fb', 'dis', 'bynd', 'nflx', 'jnug', 'ge', 'rad', 'sq', 'atvi', 'uso',
       'twtr', 'amc', 'bb', 'nok', 'pltr', 'gme']

w = [23, 54, 84, 115, 145, 176]

clusters = []
noise = []
   
nodes = []
edges = []
 
for i in w:
    
    features = pd.read_csv(path + 'week_{}.csv'.format(i), index_col = 0)

    features = features.sample(10_000) #prende ogni volta delle sample diverse
    features.reset_index(drop = True, inplace = True)
    
    features.to_csv(savedata + 'features_sample_{}.csv'.format(i))
    
    M = features.drop(columns = ['author'] + tickers)
    
    pca = PCA(n_components = 10)
    X = pca.fit(M).transform(M)         
    
    dis = pairwise_distances(X) 
    s = np.exp(-dis**2)
    np.fill_diagonal(s, 0)
    
    s[s < 0.3] = 0
    
    g = nx.from_numpy_matrix(s)   
    
    isolates = list(nx.isolates(g))    
    conn = nx.number_connected_components(g)
        
    l = [len(c) for c in sorted(nx.connected_components(g), key=len, reverse=True)]
    
    #add to noise ponts from connected components with less than 5 nodes
    cc = [c for c in sorted(nx.connected_components(g), key=len, reverse=True)]
    
    labels = []
    
    for comp in cc:
        if len(comp) < 5:
            isolates.extend(list(comp))
        else:
            labels.append(list(comp))
            
    labs = pd.Series(labels)
    labs.to_csv(savedata + 'connected_comp_labels_week_{}.csv'.format(i))
    
    isolates = list(set(isolates))
    
    g.remove_nodes_from(isolates)
    
    noise.append(len(isolates))
    
    f = [k for k in l if k >= 5] 
    
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    
    nodes.append(n_nodes)
    edges.append(n_edges)
    
    clusters.append(f)
  
    
cltrs = pd.Series(clusters)
cltrs.to_csv(savedata + 'cluster_sizes.csv')

nse = pd.Series(noise)
nse.to_csv(savedata + 'noise.csv')
    
df = pd.DataFrame(columns = ['nodes', 'edges'])
df['nodes'] = nodes
df['edges'] = edges
df.to_csv(savedata + 'nodes_edges.csv')
