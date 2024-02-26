import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-talk")
import seaborn as sns

#dbsdata = '/mnt/TANK4TB/User_feature_analysis/User_features_analysis_new/Data/With_user_filter_II/PCA_same_samples_final/DBSCAN/eps_1/'
dbsdata = '/mnt/TANK4TB/User_feature_analysis/User_features_analysis_new/Data/With_user_filter_II/PCA_same_samples_final/DBSCAN/eps_1/min_samples_15/'
saveplot = '/mnt/TANK4TB/User_feature_analysis/User_features_analysis_new/Plot/Paper_figures/'

weeks = ['24-30 Aug', '24-30 Sept', '24-30 Oct', '24-30 Nov', '24-30 Dec', '24-30 Jan']

i = 176

dbs = pd.read_csv(dbsdata + 'features_sample_week_{}.csv'.format(i), index_col = 0)

counts_dbs = dbs.labels.value_counts()

counts_dbs = pd.DataFrame(counts_dbs)

counts_dbs.reset_index(inplace = True)

####################################################################################
### i tre cluster più grandi sono:
### - 0 e 1 e 2 per dbscan
### - 0 e 1 e 2 per spectral clustering
### vedere anche chi sono gli utenti nel rumore

dbs1 = dbs[dbs.labels == 0]
dbs2 = dbs[dbs.labels == 1]
dbs3 = dbs[dbs.labels == 2]

################### plot of features in top clusters

feats = list(dbs1.columns)[:16]

dbs1.drop(columns = ['author', 'labels'], inplace = True)
dbs1_m = dbs1.mean(axis = 0).values
dbs1_m = list(dbs1_m)
std1 = dbs1.std(axis = 0).values
std1 = list(std1)
dbs1_m = np.asarray(dbs1_m)
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
#ax.fill_between(feats, both1_m-std1, both1_m+std1, color = '#04E762', alpha = 0.3)
ax.plot(feats, both2_m, linewidth = 4, alpha = 1, c = '#DC0073', label="Cluster 2: 3607")
#ax.fill_between(feats, both2_m-std2, both2_m+std2, color = '#DC0073', alpha = 0.3)
ax.plot(feats, both3_m, linewidth = 4, alpha = 1, c = '#F5B700', label="Cluster 3: 364")
#ax.fill_between(feats, both3_m-std3, both3_m+std3, color = '#F5B700', alpha = 0.3)

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






#posters f7b500, active de0074, commenters 04e756
colors = ['#04e756', '#de0074', '#f7b500']

dbsfeat = dbs.drop(columns = ['author', 'comm_sent', 'post_sent', 'comm_jargon', 'post_jargon', 's_in', 's_out'])
m1 = dbsfeat.labels == 0 
m2 = dbsfeat.labels == 1
m3 = dbsfeat.labels == 2
mtot = (m1 | m2) | m3
dbsfeat = dbsfeat[mtot]
tot = pd.melt(dbsfeat, "labels", var_name = "Features")

#sns.violinplot(data=tot, x="Features", y="value", hue="labels", cut = 0, linewidth= 0.4, palette = colors)

f, ax = plt.subplots(figsize = (10,10))
ax.yaxis.grid(True, alpha = 0.5, linewidth = 0.8)
p = sns.violinplot(data=tot, x="Features", y="value", hue="labels", cut = 0, linewidth= 0.5, palette = colors, saturation = 1, 
               scale = 'width', inner = 'box')

# sns.violinplot(data=tot, x="Features", y="value", hue="labels", order = ['num_comm','comm_score','first_children','comm_entropy','num_post','post_score','post_comms','post_entropy','k_in','k_out'],
#                hue_order = ['0', '1', '2'], dodge = True, palette = colors)

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)

# increase tick width
ax.tick_params(width=2)
ax.set_yscale('symlog')
ax.set_xlabel('Features', fontsize = 20)
ax.set_ylabel('Standardized values', fontsize = 20)
ax.tick_params(axis='x', labelsize = 15, rotation = 45)
ax.tick_params(axis='y', labelsize = 15)
# Improve the legend
p.legend(handles=p.legend_.legendHandles, labels=['Commenters', 'Active', 'Posters'], fontsize = 15, frameon = False, ncol = 3, loc = (0.04, 0.94))
#ax.legend(p, loc = (0.04, 0.94), ncol = 3, fontsize = 15, frameon = False, title = None, columnspacing = 1, handletextpad = 0)
#sns.move_legend(ax, loc = 'upper left', ncol = 3, fontsize = 15, frameon = False, title = None, columnspacing = 1, handletextpad = 0)
plt.tight_layout()
f.savefig(saveplot + 'avg_features_clusters_violin_boxplot.svg')

#######################################################################################

matplotlib.rcParams['lines.solid_capstyle'] = 'round'

f, ax = plt.subplots(figsize = (22,8))
ax.axhline(0, color = 'black', linewidth = '1', ls= '--')

#sns.despine(bottom = True, left = True)

# Show each observation with a scatterplot
# sns.stripplot(
#     data = totals, x = "Values", y = "Feature", hue = "Group",
#     dodge = True, alpha = .25, zorder = 1, legend=False)

# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
p = sns.pointplot(
    #data = tot, x = "value", y = "Features", hue = "labels", 
    data = tot, x = "Features", y = "value", hue = "labels",
    order = ['num_comm','comm_score','first_children','comm_entropy','num_post','post_score','post_comms','post_entropy','k_in','k_out'],
    hue_order = [0, 1, 2],
    join = False, dodge = .8 - .8 / 6, palette = colors,
    markers = "o", scale = 0.8, errorbar='sd', errwidth = 3, capsize=0)  #standard error

ax.set_yscale('symlog')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)

# increase tick width
ax.tick_params(width=2)

y_ticks = np.append(ax.get_yticks(), 1)
y_ticks = np.append(-10, y_ticks)

# Set xtick locations to the values of the array `x_ticks`
ax.set_yticks(y_ticks)
#ax.set_xlim(-3, 2)

ax.set_xlabel('Features', fontsize = 20)
ax.set_ylabel('Standardized values', fontsize = 20)
ax.tick_params(axis='x', labelsize = 20, rotation = 45)
ax.tick_params(axis='y', labelsize = 20)
ax.yaxis.grid(True, alpha = 0.5, linewidth = 0.8)
# Improve the legend
p.legend(handles=p.legend_.legendHandles, labels=['Commenters', 'Active', 'Posters'], fontsize = 20, frameon = False, ncol = 3, loc = (0, 1.02))
#sns.move_legend(ax, loc = 'upper left', ncol = 1, fontsize = 15, frameon = False, title = None, columnspacing = 1, handletextpad = 0)
plt.tight_layout()
f.savefig(saveplot + 'avg_features_clusters_pointplot_new.pdf', dpi = 300)









# f, ax = plt.subplots(figsize = (10,10))
# ax.axvline(0, color = 'red', linewidth = '2')

# #sns.despine(bottom = True, left = True)

# # Show each observation with a scatterplot
# # sns.stripplot(
# #     data = totals, x = "Values", y = "Feature", hue = "Group",
# #     dodge = True, alpha = .25, zorder = 1, legend=False)

# # Show the conditional means, aligning each pointplot in the
# # center of the strips by adjusting the width allotted to each
# # category (.8 by default) by the number of hue levels
# sns.stripplot(
#     data = tot, x = "value", y = "Features", hue = "labels",
#     dodge=True, alpha=.2, legend=False, palette = colors)

# sns.pointplot(
#     data = tot, x = "value", y = "Features", hue = "labels",
#     order = ['num_comm','comm_score','first_children','comm_entropy','num_post','post_score','post_comms','post_entropy','k_in','k_out'],
#     hue_order = [0, 1, 2],
#     join = False, dodge = .8 - .8 / 6, palette = colors,
#     markers = "_", scale = 0.8, errorbar=None)
    

# ax.set_xscale('symlog')
# for axis in ['top','bottom','left','right']:
#     ax.spines[axis].set_linewidth(2)

# # increase tick width
# ax.tick_params(width=2)

# #ax.set_xlim(-3, 2)

# ax.set_xlabel('Standardized values', fontsize = 20)
# ax.set_ylabel('Features', fontsize = 20)
# ax.tick_params(axis='x', labelsize = 20)
# ax.tick_params(axis='y', labelsize = 20)
# ax.xaxis.grid(True, alpha = 0.5, linewidth = 0.8)
# # Improve the legend
# p.legend(handles=p.legend_.legendHandles, labels=['Commenters', 'Active', 'Posters'], fontsize = 20, frameon = False, ncol = 3, loc = (0, 1.02))
# #sns.move_legend(ax, loc = 'upper left', ncol = 1, fontsize = 15, frameon = False, title = None, columnspacing = 1, handletextpad = 0)
# plt.tight_layout()
# f.savefig(saveplot + 'avg_features_clusters_pointplot.pdf', dpi = 300)




