import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-talk")
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

tickers = ['spy',
       'amd', 'tsla', 'mu', 'aapl', 'amzn', 'msft', 'snap', 'nvda', 'spce',
       'fb', 'dis', 'bynd', 'nflx', 'jnug', 'ge', 'rad', 'sq', 'atvi', 'uso',
       'twtr', 'amc', 'bb', 'nok', 'pltr', 'gme']

saveplot = ''
path = ''
savedata = ''


w = [23, 54, 84, 115, 145, 176]

weeks = ['24-30 Aug', '24-30 Sept', '24-30 Oct', '24-30 Nov', '24-30 Dec', '24-30 Jan']

silscore = []
j = 0
for i in w:
        
    features = pd.read_csv(path + 'features_sample_{}.csv'.format(i), index_col = 0)
    features.drop(columns = tickers, inplace = True)
    
    labels = pd.read_csv(path + 'connected_comp_labels_week_{}.csv'.format(i), index_col = 0)

    feat = features.drop(columns = ['author'])
    
    pca = PCA(n_components = 10)
    X = pca.fit(feat).transform(feat)
    
    projection = TSNE().fit_transform(X)
    
    features['labels'] = -1
    
    for ind in range(0, len(labels)):
        
        labs = labels.loc[ind][0]
        labs = labs[1:-1]
        labs = labs.split(', ')
        labs = [int(l) for l in labs]
        
        m = features.index.isin(labs)
        #features[m].labels = ind
        features.loc[m, 'labels'] = ind
        
    features.to_csv(savedata + 'features_sample_final_{}.csv'.format(i))
    
    clusters = len(features.labels.unique())-1
    noise = len(features[features.labels == -1]) 
    
    df = features[features.labels != -1]
    labs2 = df.labels
    df.drop(columns = ['labels', 'author'], inplace = True)
    pca2 = PCA(n_components = 10)
    X2 = pca2.fit(df).transform(df)

    silhouette = silhouette_score(X2, labs2)
    silscore.append(silhouette)
    
    color_palette = sns.color_palette('Paired', clusters)
    
    cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in features.labels]
    
    fig, ax = plt.subplots(figsize = (10,10))
    plt.scatter(*projection.T, s=50, linewidth=0, c=cluster_colors, alpha=0.25)
    plt.title('{}, clusters: {}, noise: {:.2f}%'.format(weeks[j], clusters, noise/len(features)*100), fontsize = 20)
    fig.savefig(saveplot + "week_{}.png".format(i))
    
    j += 1
    
silscore = pd.Series(silscore)
silscore.to_csv(savedata + 'silhouette.csv')
