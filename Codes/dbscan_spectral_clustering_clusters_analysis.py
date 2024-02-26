import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-talk")
import seaborn as sns
from matplotlib_venn import venn2, venn2_circles
from matplotlib import pyplot as plt
from math import pi
from sklearn.metrics.cluster import adjusted_rand_score

scdata = ''
dbsdata = ''

saveplot = ''
savedata = ''

weeks = ['24-30 Aug', '24-30 Sept', '24-30 Oct', '24-30 Nov', '24-30 Dec', '24-30 Jan']

i = 176

dbs = pd.read_csv(dbsdata + 'features_sample_week_{}.csv'.format(i), index_col = 0)
sc = pd.read_csv(scdata + 'features_sample_final_{}.csv'.format(i), index_col = 0)

counts_dbs = dbs.labels.value_counts()
counts_sc = sc.labels.value_counts()

counts_dbs = pd.DataFrame(counts_dbs)
counts_sc = pd.DataFrame(counts_sc)

counts_sc.reset_index(inplace = True)
counts_dbs.reset_index(inplace = True)

bar_width = 0.7

counts_sc['index'] = counts_sc['index'].astype('string')
counts_dbs['index'] = counts_dbs['index'].astype('string')

fig, ax = plt.subplots(2, figsize = (10,10), sharex = False, sharey = True)
ax[0].bar(counts_sc['index'], counts_sc.labels, bar_width, color = '#F7B05B', capstyle = 'round', joinstyle = 'round', edgecolor = "#F7B05B", linewidth = 5, label = 'Spectral clustering')
ax[1].bar(counts_dbs['index'], counts_dbs.labels, bar_width, color = '#F75C71', capstyle = 'round', joinstyle = 'round', edgecolor = "#F75C71", linewidth = 5, label = 'DBSCAN')

ax[0].set_xlabel('Clusters', fontsize = 20)
ax[0].set_ylabel('Counts', fontsize = 20)
ax[0].tick_params(axis='x', labelsize = 15)
ax[0].tick_params(axis='y', labelsize = 15)
ax[0].set_yscale('log')
#ax.set_xscale('log')
ax[0].spines['right'].set_visible(True)
ax[0].spines['top'].set_visible(True)
ax[0].legend(fontsize = 20, frameon = False)

ax[1].set_xlabel('Clusters', fontsize = 20)
ax[1].set_ylabel('Counts', fontsize = 20)
ax[1].tick_params(axis='x', labelsize = 15)
ax[1].tick_params(axis='y', labelsize = 15)
ax[1].set_yscale('log')
#ax.set_xscale('log')
ax[1].spines['right'].set_visible(True)
ax[1].spines['top'].set_visible(True)
ax[1].legend(fontsize = 20, frameon = False)

plt.suptitle('Cluster sizes week {}'.format(weeks[5]), fontsize = 20)
#plt.ylim(0.00001, 10**6)
plt.tight_layout()
fig.savefig(saveplot+"cluster_sizes_week_{}.png".format(i))

####################################################################################
### i tre cluster più grandi sono:
### - 0 e 1 e 2 per dbscan
### - 0 e 1 e 2 per spectral clustering
### vedere anche chi sono gli utenti nel rumore

dbs1 = dbs[dbs.labels == 0]
dbs2 = dbs[dbs.labels == 1]
dbs3 = dbs[dbs.labels == 2]

sc1 = sc[sc.labels == 0]
sc2 = sc[sc.labels == 1]
sc3 = sc[sc.labels == 2]

m1 = dbs1.author.isin(sc1.author) #utenti in dbs1 che sono anche in sc1
dbs1_only = dbs1[~m1]

n1 = sc1.author.isin(dbs1.author) #utenti in sc1 che sono anche in dbs1
sc1_only = sc1[~n1]

both1 = dbs1[m1]
both1.reset_index(drop = True, inplace = True)
#both12 = sc1[n1]

m2 = dbs2.author.isin(sc2.author) #utenti in dbs2 che sono anche in sc2
dbs2_only = dbs2[~m2]

n2 = sc2.author.isin(dbs2.author) #utenti in sc2 che sono anche in dbs2
sc2_only = sc2[~n2]

both2 = dbs2[m2]
both2.reset_index(drop = True, inplace = True)
#both2 = sc2[n2]

m3 = dbs3.author.isin(sc3.author) #utenti in dbs1 che sono anche in sc1
dbs3_only = dbs3[~m3]

n3 = sc3.author.isin(dbs3.author) #utenti in sc1 che sono anche in dbs1
sc3_only = sc3[~n3]

both3 = dbs3[m3]
both3.reset_index(drop = True, inplace = True)

both1.to_csv(savedata + 'cluster_1.csv')
both2.to_csv(savedata + 'cluster_2.csv')
both3.to_csv(savedata + 'cluster_3.csv')

### venn diagram for top groups

colors = ['#F75C71', '#F7B05B']

fig, ax = plt.subplots(figsize = (10,10))
v = venn2(subsets = (len(dbs1_only), len(sc1_only), len(both1)), set_labels = ('DBSCAN', 'Spectral clustering'), set_colors = colors, alpha = 0.6)
#c = venn2_circles(subsets = (len(dbs1_only), len(sc1_only), len(both1)), linestyle='-', linewidth=2, color="black")

for text in v.subset_labels:
  text.set_color('black')
  text.set_fontsize(16)
  #text.set_fontweight('bold')
  
for text in v.set_labels:
  text.set_fontsize(18)
  
plt.title('Top 1 cluster for DBSCAN and spectral clustering', fontsize = 18)
plt.tight_layout()
plt.show()
fig.savefig(saveplot+"venn_top1_week_{}.png".format(i))



fig, ax = plt.subplots(figsize = (10,10))
v = venn2(subsets = (len(dbs2_only), len(sc2_only), len(both2)), set_labels = ('DBSCAN', 'Spectral clustering'), set_colors = colors, alpha = 0.6)

for text in v.subset_labels:
  text.set_color('black')
  text.set_fontsize(16)
  
for text in v.set_labels:
  text.set_fontsize(18)
  
plt.title('Top 2 cluster for DBSCAN and spectral clustering', fontsize = 18)
plt.tight_layout()
plt.show()
fig.savefig(saveplot+"venn_top2_week_{}.png".format(i))


fig, ax = plt.subplots(figsize = (10,10))
v = venn2(subsets = (len(dbs3_only), len(sc3_only), len(both3)), set_labels = ('DBSCAN', 'Spectral clustering'), set_colors = colors, alpha = 0.6)

for text in v.subset_labels:
  text.set_color('black')
  text.set_fontsize(16)
  
for text in v.set_labels:
  text.set_fontsize(18)
  
plt.title('Top 3 cluster for DBSCAN and spectral clustering', fontsize = 18)
plt.tight_layout()for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)

# increase tick width
ax.tick_params(width=2)
plt.show()
fig.savefig(saveplot+"venn_top3_week_{}.png".format(i))


#### radarplot for features 0f top 3 groups

feats = list(dbs1.columns)
feats = feats[:16]

#both1 = both1[feats]

both1.drop(columns = ['author', 'labels'], inplace = True)
both1_m = both1.mean(axis = 0).values
both1_m = list(both1_m)
both1_m.append(both1_m[0])

#both2 = both2[feats]

both2.drop(columns = ['author', 'labels'], inplace = True)
both2_m = both2.mean(axis = 0).values
both2_m = list(both2_m)
both2_m.append(both2_m[0])


both3.drop(columns = ['author', 'labels'], inplace = True)
both3_m = both3.mean(axis = 0).values
both3_m = list(both3_m)
both3_m.append(both3_m[0])


dbs1_m = dbs1.mean(axis = 0).values
dbs1_m = list(dbs1_m)
dbs1_m.append(dbs1_m[0])

sc1_m = sc1.mean(axis = 0).values
sc1_m = list(sc1_m)
sc1_m.append(sc1_m[0])


dbs2_m = dbs2.mean(axis = 0).values
dbs2_m = list(dbs2_m)
dbs2_m.append(dbs2_m[0])

sc2_m = sc2.mean(axis = 0).values
sc2_m = list(sc2_m)
sc2_m.append(sc2_m[0])


dbs3_m = dbs3.mean(axis = 0).values
dbs3_m = list(dbs3_m)
dbs3_m.append(dbs3_m[0])

sc3_m = sc3.mean(axis = 0).values
sc3_m = list(sc3_m)
sc3_m.append(sc3_m[0])

 
# number of variable
categories = feats
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
fig = plt.figure(figsize = (10,10))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories)
ax.set_rlabel_position(0)

# Ind1
ax.plot(angles, both1_m, linewidth=3, linestyle='solid', c = '#04E762', label = "Cluster 1: 5058")
 
# Ind2
ax.plot(angles, both2_m, linewidth=3, linestyle='solid', c = '#DC0073', label="Cluster 2: 3607")

# Ind3
ax.plot(angles, both3_m, linewidth=3, linestyle='solid', c = '#F5B700', label="Cluster 3: 364")

plt.title('Top clusters overlapping elements 24-30 Jan', fontsize = 15)
plt.legend(loc='upper right', bbox_to_anchor=(0.01, 0.1), frameon = False, fontsize = 15)
plt.tight_layout()
plt.show()
fig.savefig(saveplot + "top_3_clusters_radar_week_{}.png".format(i)) 


#####################################
### compute the adjusted rand score for the clustering

ari = adjusted_rand_score(dbs.labels, sc.labels)

################### plot of features in top clusters

feats = list(dbs1.columns)[:16]

both1.drop(columns = ['author', 'labels'], inplace = True)
both1_m = both1.mean(axis = 0).values
both1_m = list(both1_m)
std1 = both1.std(axis = 0).values
std1 = list(std1)
both1_m = np.asarray(both1_m)
std1 = np.asarray(std1)

both2.drop(columns = ['author', 'labels'], inplace = True)
both2_m = both2.mean(axis = 0).values
both2_m = list(both2_m)
std2 = both2.std(axis = 0).values
std2 = list(std2)
both2_m = np.asarray(both2_m)
std2 = np.asarray(std2)

both3.drop(columns = ['author', 'labels'], inplace = True)
both3_m = both3.mean(axis = 0).values
both3_m = list(both3_m)
std3 = both3.std(axis = 0).values
std3 = list(std3)
both3_m = np.asarray(both3_m)
std3 = np.asarray(std3)

fig, ax = plt.subplots(figsize = (10,10))

ax.plot(feats, both1_m, linewidth = 4, alpha = 1, c = '#04E762', label = "Cluster 1: 5058")
ax.plot(feats, both2_m, linewidth = 4, alpha = 1, c = '#DC0073', label="Cluster 2: 3607")
ax.plot(feats, both3_m, linewidth = 4, alpha = 1, c = '#F5B700', label="Cluster 3: 364")

ax.set_xlabel('Features', fontsize = 30)
ax.set_ylabel('Avereage value', fontsize = 30)
ax.tick_params(axis='x', labelsize = 25, rotation = 90)
ax.tick_params(axis='y', labelsize = 25)
ax.set_yscale('symlog')
#ax.set_xscale('log')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)

# increase tick width
ax.tick_params(width=2)
ax.spines['right'].set_visible(True)
ax.spines['top'].set_visible(True)
ax.legend(fontsize = 30, frameon = False)
ax.set_title('Top 3 clusters', fontsize = 30)
plt.tight_layout()
plt.grid(alpha = 0.2)
fig.savefig(saveplot + "top_3_clusters_features_plot_2.svg")




