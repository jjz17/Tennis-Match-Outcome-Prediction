import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

data = pd.read_csv(f'..{os.path.sep}data{os.path.sep}wrangled_data.csv', index_col=0)
# url = 'https://raw.githubusercontent.com/jjz17/Tennis-Match-Outcome-Prediction/main/data/wrangled_data.csv'
# data = pd.read_csv(url, index_col=0)

# Data visualization
plot_data = data[['p1_win_fs', 'p1_win']]

plot_data.groupby('p1_win_fs').p1_win.value_counts().unstack(0).plot.barh()
plt.show()

# %%
sns.catplot(x='p1_win', y='fs_s1_points', kind='box', data=data)
plt.title('Distribution of Player 1 Points for win vs loss of match')
plt.xlabel('Player 1 Win')
plt.ylabel('Points Scored')
plt.show()
# %%
sns.catplot(x='p1_win', y='fs_s2_points', kind='box', data=data)
plt.title('Distribution of Player 2 Points for win vs loss of match')
plt.xlabel('Player 1 Win')
plt.ylabel('Points Scored')
plt.show()

# %%

rel_columns = ['fs_s1_momentum', 'fs_s2_momentum',
               'fs_s1_breaks', 'fs_s2_breaks', 'fs_s1_aces',
               'fs_s2_aces', 'fs_s1_points', 'fs_s2_points', 'p1_win_fs', 'p1_win']

ml_data = data[rel_columns].reset_index()

sns.heatmap(ml_data.corr())
plt.show()

#%%
most_important = ['fs_s1_momentum', 'fs_s2_momentum', 'fs_s1_points', 'fs_s2_points', 'p1_win_fs', 'p1_win']
less_data = data[most_important].reset_index()
# sns.pairplot(less_data)
sns.pairplot(less_data)
plt.show()

sns.relplot(x='fs_s1_points', y='fs_s2_points', hue='p1_win', data=less_data)
plt.show()

sns.relplot(x='fs_s1_momentum', y='fs_s2_momentum', hue='p1_win', data=less_data)
plt.show()

sns.relplot(x='fs_s1_points', y='fs_s1_momentum', hue='p1_win', data=less_data)
plt.show()

sns.relplot(x='fs_s2_points', y='fs_s2_momentum', hue='p1_win', data=less_data)
plt.show()