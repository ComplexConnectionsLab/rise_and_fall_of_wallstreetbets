import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-talk")
#from sklearn import metrics
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
    
#################################################################################################################################################
########### using same samples used for spectral clustering

path = '/mnt/TANK4TB/User_feature_analysis/User_features_analysis_new/Data/With_user_filter_II/PCA_same_samples_final/Spectral_Clustering/s_lt_0.3/'
savepath = '/mnt/TANK4TB/User_feature_analysis/User_features_analysis_new/Data/With_user_filter_II/PCA_same_samples_final/DBSCAN/eps_1/min_samples_15/'
#saveplot = '/mnt/TANK4TB/User_feature_analysis/User_features_analysis_new/Plot/With_user_filter_II/PCA/PCA_same_samples_final/DBSCAN/'
saveplot = '/mnt/TANK4TB/User_feature_analysis/User_features_analysis_new/Plot/Paper_figures/DBSCAN_tsne/DPI_500/'

w = [23, 54, 84, 115, 145, 176]

weeks = ['24-30 Aug', '24-30 Sept', '24-30 Oct', '24-30 Nov', '24-30 Dec', '24-30 Jan']

colors = ['#04e756', '#de0074', '#f7b500', '#ADA1CE', '#388D72', '#00FFF2', '#F2FF00', '#A600FF', '#F6DBFF', '#77685D']
#posters f7b500, active de0074, commenters 04e756

#silscore = []
j = 0
for i in w:
        
    features = pd.read_csv(path + 'features_sample_final_{}.csv'.format(i), index_col = 0)
    features.drop(columns = ['labels'], inplace = True)

    feat = features.drop(columns = ['author'])
    
    pca = PCA(n_components = 10)
    X = pca.fit(feat).transform(feat)
    
    projection = TSNE().fit_transform(X)
    
    db = DBSCAN(eps=1, min_samples=15).fit(X)
    
    #features = pd.read_csv(savepath + 'features_sample_week_{}.csv'.format(i), index_col = 0)
    l = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise_ = list(db.labels_).count(-1)
    
    features['labels'] = db.labels_
    features.to_csv(savepath + 'features_sample_week_{}.csv'.format(i))
    
    n_noise = list(features.labels).count(-1)
    l = features.labels.nunique() - 1
    
    # df = features[features.labels != -1]
    # labs2 = df.labels
    #df = features.drop(columns = ['labels', 'author'])
    #X = df.to_numpy()
    #projection = TSNE().fit_transform(X)
    
    # pca2 = PCA(n_components = 10)
    # X2 = pca2.fit(df).transform(df)
    
    #silhouette = silhouette_score(X2, labs2)
    #silscore.append(silhouette)
         
    csize = pd.DataFrame(features['labels'].value_counts()).reset_index()
    order = list(csize['index'])
    order.remove(-1)
    
    
    color_palette = [colors[i] for i in order]
    
    cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in features.labels]
    
    fig, ax = plt.subplots(facecolor='none', figsize = (10,10))
    plt.scatter(*projection.T, s=50, linewidth=0, c=cluster_colors, alpha=0.25)
    
    plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
    plt.axis('off')
    plt.tight_layout()
    fig.savefig(saveplot + "dbscan_clustering_no-axis_week_{}.png".format(i), dpi = 500, transparent = True)
    
    print('{}, clusters: {}, noise: {:.2f}%'.format(weeks[j], l, n_noise/len(features)*100))
    
    j += 1

# silscore = pd.Series(silscore)
# silscore.to_csv(savepath + 'silhouette.csv')






