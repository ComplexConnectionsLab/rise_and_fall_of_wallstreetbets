import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-talk")
from sklearn.metrics.cluster import adjusted_rand_score

scdata = ''
dbsdata = ''
saveplot = ''

w = [23, 54, 84, 115, 145, 176]

weeks = ['24-30 Aug', '24-30 Sept', '24-30 Oct', '24-30 Nov', '24-30 Dec', '24-30 Jan']

ari = []

for i in w:
    
    dbs = pd.read_csv(dbsdata + 'features_sample_week_{}.csv'.format(i), index_col = 0)
    sc = pd.read_csv(scdata + 'features_sample_final_{}.csv'.format(i), index_col = 0)

    aind = adjusted_rand_score(dbs.labels, sc.labels)
    ari.append(aind)

    
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(weeks, ari, linestyle = '-', marker = 'o', markersize = '15', linewidth = '8', c = '#5603AD')

ax.set_xlabel('Week', fontsize = 20)
ax.set_ylabel('Adjusted Rand Index', fontsize = 20)
ax.tick_params(axis='x', labelsize = 15, rotation = 45)
ax.tick_params(axis='y', labelsize = 15)
ax.spines['right'].set_visible(True)
ax.spines['top'].set_visible(True)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)

# increase tick width
ax.tick_params(width=2)
plt.tight_layout()
ax.set_ylim(0,1)
fig.savefig(saveplot+"ARI_vs_t.png", dpi = 300)
