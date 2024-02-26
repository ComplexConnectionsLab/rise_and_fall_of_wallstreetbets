import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import networkx as nx
import numpy.linalg
import matplotlib
matplotlib.style.use("seaborn-talk")
from sklearn import metrics
#from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances


#savedata = '/media/lorenzo/Backup4TBU3/User_feature_analysis/User_features_analysis_new/Data/With_user_filter_II/PCA_spectral_clustering/Same_sample_size/'
path = '/mnt/TANK4TB/User_feature_analysis/User_features_analysis_new/Data/With_user_filter_II/User_standardized_features_new/'
#savepath = '/media/lorenzo/Backup4TBU3/User_feature_analysis/User_features_analysis_new/Plot/With_user_filter_II/PCA/PCA_spectral_clustering/Same_sample_size/Graphs/No_noise/'
savedata = '/mnt/TANK4TB/User_feature_analysis/User_features_analysis_new/Data/With_user_filter_II/PCA_ensamble_avg_new/Spectral_clustering/'

#range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

dropcols = ['spy','amd','tsla','mu','aapl','amzn','msft','snap','nvda','spce','fb','dis','bynd','nflx','jnug','ge','rad','sq','atvi','uso','twtr','amc','bb','nok','pltr','gme']


w = [23, 54, 84, 115, 145, 176]

cols = list(np.arange(0,100))
cols = list(map(str, cols))

df = pd.DataFrame(0, index=np.arange(len(w)), columns=cols)
nf = pd.DataFrame(0, index=np.arange(len(w)), columns=cols)
cf = pd.DataFrame(0, index=np.arange(len(w)), columns=cols)

#readpath = '/media/lorenzo/Backup4TBU3/User_feature_analysis/User_features_analysis_new/Data/With_user_filter_II/PCA_spectral_clustering/Same_sample_size/'

for j in range(0,100):
    
    clusters = []
    clus_numb = []
    noise = []
       
    #nodes = []
    #edges = []
     
    for i in w:
        
        features = pd.read_csv(path + 'week_{}.csv'.format(i), index_col = 0)
        features.drop(columns = dropcols, inplace = True)
    
        features = features.sample(10_000) #prende ogni volta delle sample diverse
        features.reset_index(drop = True, inplace = True)
        
        #features.to_csv(savedata + 'features_sample_{}.csv'.format(i))
        
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
        
        # labels = []
        
        for comp in cc:
            if len(comp) < 5:
                isolates.extend(list(comp))
        #     else:
        #         labels.append(list(comp))
                
        # labs = pd.Series(labels)
        # labs.to_csv(savedata + 'connected_comp_labels_week_{}.csv'.format(i))
        
        isolates = list(set(isolates))
        
        g.remove_nodes_from(isolates)
        
        noise.append(len(isolates))
        
        f = [k for k in l if k >= 5]
        
        # fig = plt.figure(figsize=(10, 10))
        # nx.draw_networkx(g, node_size = 10, alpha = 0.8, width = 1, node_color = 'red', with_labels = False)
        # plt.tight_layout()
        # plt.show()
        # fig.savefig(savepath + "graph_week_{}.png".format(i))   
        
        #n_nodes = g.number_of_nodes()
        #n_edges = g.number_of_edges()
        
        #nodes.append(n_nodes)
        #edges.append(n_edges)
        
        clusters.append(f)
        clus_numb.append(len(f))
    
    
    df['{}'.format(j)] = clus_numb
    nf['{}'.format(j)] = noise
    cf['{}'.format(j)] = clusters   
    
    
df.to_csv(savedata + 'cluster_numbers.csv') 
nf.to_csv(savedata + 'noise.csv')
cf.to_csv(savedata + 'cluster_sizes.csv')   