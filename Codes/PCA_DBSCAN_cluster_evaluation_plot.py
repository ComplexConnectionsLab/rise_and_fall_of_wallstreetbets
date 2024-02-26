import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-talk")
import matplotlib.ticker as mticker

path = ''
savepath = ''

week = np.arange(0,178)
settimane = ['01/08-07/08', '20/09-26/09', '9/11-15/11', '29/12-04/01']

epsval = [0.5, 1, 1.5]
minsampval = [5, 10, 15]

l = 1
fig, ax = plt.subplots(3,3, figsize = (15,10), sharex = True, sharey = True)

for i in range(0,3):
    for j in range(0,3):
        
        clusters = pd.read_csv(path + '{}/cluster_numbers.csv'.format(l), index_col = 0)
        noise = pd.read_csv(path + '{}/noisepoints.csv'.format(l), index_col = 0)
        sizes = pd.read_csv(path + '{}/cluster_sizes.csv'.format(l), index_col = 0)
        db_index = pd.read_csv(path + '{}/index.csv'.format(l), index_col = 0)
        noise = noise/10_000

        cluster_mean = clusters.mean(axis = 1)
        cluster_std = clusters.std(axis = 1)#/np.sqrt(100)
        
        noise_mean = noise.mean(axis = 1)
        noise_std = noise.std(axis = 1)#/np.sqrt(100)
        
        size_mean = sizes.mean(axis = 1)
        size_std = sizes.std(axis = 1)#/np.sqrt(100)
        
        index_mean = db_index.mean(axis = 1)
        index_std = db_index.std(axis = 1)#/np.sqrt(100)

        
        ax[i][j].plot(week, noise_mean.rolling(7).mean(), linewidth = 4, ls='-', c = 'blue', label = 'noise', solid_capstyle='round')
        ax[i][j].fill_between(week, noise_mean.rolling(7).mean()-noise_std.rolling(7).mean(), noise_mean.rolling(7).mean()+noise_std.rolling(7).mean(), alpha=0.2, edgecolor='blue', facecolor='blue', antialiased=True)

        ax2 = ax[i][j].twinx()
        ax2.plot(week, index_mean.rolling(7).mean(), linewidth = 4, ls='-', c = 'red', label = 'D-B index', solid_capstyle='round')
        ax2.fill_between(week, index_mean.rolling(7).mean()-index_std.rolling(7).mean(), index_mean.rolling(7).mean()+index_std.rolling(7).mean(), alpha=0.2, edgecolor='red', facecolor='red', antialiased=True)
        
        ax2.set_ylim(0.2,1.5)
        
        if ((j == 0) or (j == 1)):
            ax2.yaxis.set_tick_params(labelright=False)
            
        if (i == 2):
            ax[i][j].set_xlabel('Week', fontsize = 20)
            #ticks_loc = ax[i][j].get_xticks().tolist()
            #ticks_loc = ticks_loc[1:5]
            ticks_loc = [0, 50, 100, 150]
            ax[i][j].xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            ax[i][j].set_xticklabels(settimane, rotation = 45)
            
        if (j == 2):
            ax2.set_ylabel('D-B index', fontsize = 20)

        if (j == 0):
            ax[i][j].set_ylabel('Fraction of noise', fontsize = 20)
            
        ax2.legend(loc = 'upper right', frameon = False)
        ax[i][j].legend(loc = 'upper left', frameon = False)
        
        ax[i][j].set_title('eps = {}, min_samples = {}'.format(epsval[i], minsampval[j]))
        l += 1
plt.tight_layout()
plt.savefig(savepath + 'DBSCAN_cluster_evaluation.pdf', dpi = 300)




