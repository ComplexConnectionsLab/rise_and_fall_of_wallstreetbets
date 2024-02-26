import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("seaborn-talk")
import matplotlib.ticker as mticker

path = '/mnt/TANK4TB/User_feature_analysis/User_features_analysis_new/Data/With_user_filter_II/PCA_cluster_evaluation/D-B_index/'
savepath = '/mnt/TANK4TB/User_feature_analysis/User_features_analysis_new/Plot/Paper_figures/'

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
            
        # ax[i][j].set_xlabel('Week', fontsize = 15)
        # ax[i][j].set_ylabel('Fraction of noise', fontsize = 15)
        # ax[i][j].tick_params(axis='x', labelsize = 10, rotation = 45)
        # ax[i][j].tick_params(axis='y', labelsize = 10)
        # #ax[i][j].set_yscale('log')
        
        # #ax2.set_xlabel('Week', fontsize = 30)
        # ax2.set_ylabel('D-B index', fontsize = 15, rotation = 270)
        # ax2.tick_params(axis='x', labelsize = 15, rotation = 45)
        # ax2.tick_params(axis='y', labelsize = 10)
        # #ax2.legend(fontsize = 10, frameon = False, loc = (1.1, 0.57))
        # #ax2.yaxis.set_label_coords(1.08, .5)
        
        # for axis in ['top','bottom','left','right']:
        #     ax[i][j].spines[axis].set_linewidth(2)
        
        # # increase tick width
        # ax[i][j].tick_params(width=2)
        
        # # ticks_loc = ax.get_xticks().tolist()
        # # ticks_loc = ticks_loc[1:9]
        # # ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        # # ax.set_xticklabels(settimane)
        
        # ax[i][j].spines['right'].set_visible(True)
        # ax[i][j].spines['top'].set_visible(True)
        #ax[i][j].legend(fontsize = 10, frameon = False, loc = (1.1, 0.67))
        ax2.legend(loc = 'upper right', frameon = False)
        ax[i][j].legend(loc = 'upper left', frameon = False)
        
        ax[i][j].set_title('eps = {}, min_samples = {}'.format(epsval[i], minsampval[j]))
        l += 1
#ax[0].set_title('Spectral clustering', fontsize = 20)ù


#plt.legend(fontsize = 20)
plt.tight_layout()
plt.savefig(savepath + 'DBSCAN_cluster_evaluation.pdf', dpi = 300)




