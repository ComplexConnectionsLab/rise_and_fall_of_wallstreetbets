import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.style.use("seaborn-talk")
from tqdm import tqdm


path = '/mnt/TANK4TB/User_feature_analysis/User_features_analysis_new/Data/With_user_filter_II/User_standardized_features_new/'
savedata = '/mnt/TANK4TB/User_feature_analysis/User_features_analysis_new/Data/With_user_filter_II/PCA_cluster_evaluation/D-B_index/'
#savedata = '/mnt/TANK4TB/User_feature_analysis/User_features_analysis_new/Data/With_user_filter_II/PCA_cluster_evaluation/C-H_index/'

dropcols = ['spy','amd','tsla','mu','aapl','amzn','msft','snap','nvda','spce','fb','dis','bynd','nflx','jnug','ge','rad','sq','atvi','uso','twtr','amc','bb','nok','pltr','gme']

#w = [23, 54, 84, 115, 145, 176]
#w = [7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133, 140, 147, 154, 161, 168, 175]
w = np.arange(0,178)

cols = list(np.arange(0,100))
cols = list(map(str, cols))


#epsilon = [0.5, 1, 1.5]
epsilon = [1]
min_neigh = [5, 10, 15]

folder = 4
for epsi in tqdm(epsilon):
    for mins in min_neigh:
        
        df = pd.DataFrame(0, index=np.arange(len(w)), columns=cols)
        nf = pd.DataFrame(0, index=np.arange(len(w)), columns=cols)
        cf = pd.DataFrame(0, index=np.arange(len(w)), columns=cols)
        ch_index = pd.DataFrame(0, index=np.arange(len(w)), columns=cols)
        
        for j in range(0,100):

            noisepoints = []
            clusters = []
            sizes = []
            chind = []
            
            for i in w:
                
                features = pd.read_csv(path + 'week_{}.csv'.format(i), index_col = 0)
                features.drop(columns = dropcols, inplace = True)
                features = features.sample(10_000)
                features.reset_index(drop = True, inplace = True)
             
                
                M = features.drop(columns = ['author'])
                
                pca = PCA(n_components = 10)
                X = pca.fit(M).transform(M)
                    
                db = DBSCAN(eps = epsi, min_samples = mins).fit(X)
                
                l = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
                n_noise_ = list(db.labels_).count(-1)
                 
                noisepoints.append(n_noise_)
                clusters.append(l)
                
                count = pd.Series(db.labels_).value_counts()
                count = count[count.index != -1]
                sizes.append(list(count.values))
                
                dbsa = pd.DataFrame(X)
                dbsa['labels'] = db.labels_
                dbsa = dbsa[dbsa.labels != -1]
                dbsa.reset_index(drop = True, inplace = True)
                
                clabs = dbsa.labels
                dbsa.drop(columns = 'labels', inplace = True)
                Y = dbsa.to_numpy()
                try:
                    #chind.append(metrics.calinski_harabasz_score(Y, clabs))
                    chind.append(metrics.davies_bouldin_score(Y, clabs))
                except ValueError:
                    chind.append(np.nan)
                
                           
            df['{}'.format(j)] = clusters
            nf['{}'.format(j)] = noisepoints
            cf['{}'.format(j)] = sizes
            ch_index['{}'.format(j)] = chind
            
        
        df.to_csv(savedata + '{}/cluster_numbers.csv'.format(folder))
        nf.to_csv(savedata + '{}/noisepoints.csv'.format(folder))
        cf.to_csv(savedata + '{}/cluster_sizes.csv'.format(folder))
        ch_index.to_csv(savedata + '{}/index.csv'.format(folder))
        
        folder += 1
    
    

