import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

data = pd.read_csv(f'..{os.path.sep}data{os.path.sep}wrangled_data.csv')
# url = 'https://raw.githubusercontent.com/jjz17/Tennis-Match-Outcome-Prediction/main/data/wrangled_data.csv'
# data = pd.read_csv(url, index_col=0)

# Data visualization
plot_data = data[['s1 fs win', 's1 win']]

plot_data.groupby('s1 fs win')['s1 win'].value_counts().unstack(0).plot.barh()
plt.show()

# %%
sns.catplot(x='s1 win', y='s1 fs points', kind='box', data=data)
plt.title('Distribution of Player 1 Points for win vs loss of match')
plt.xlabel('Player 1 Win')
plt.ylabel('Points Scored')
plt.show()
# %%
sns.catplot(x='s1 win', y='s2 fs points', kind='box', data=data)
plt.title('Distribution of Player 2 Points for win vs loss of match')
plt.xlabel('Player 1 Win')
plt.ylabel('Points Scored')
plt.show()

# %%

rel_columns = ['s1 fs momentum', 's2 fs momentum',
               's1 fs breaks', 's2 fs breaks', 's1 fs aces',
               's2 fs aces', 's1 fs points', 's2 fs points', 's1 fs win', 's1 win']

ml_data = data[rel_columns].reset_index()

sns.heatmap(ml_data.corr())
plt.show()

# %%
most_important = ['s1 fs momentum', 's2 fs momentum', 's1 fs points', 's2 fs points', 's1 fs win', 's1 win']
less_data = data[most_important].reset_index()
# sns.pairplot(less_data)
sns.pairplot(less_data)
plt.show()

sns.relplot(x='s1 fs points', y='s2 fs points', hue='s1 win', data=less_data)
plt.show()

sns.relplot(x='s1 fs momentum', y='s2 fs momentum', hue='s1 win', data=less_data)
plt.show()

sns.relplot(x='s1 fs points', y='s1 fs momentum', hue='s1 win', data=less_data)
plt.show()

sns.relplot(x='s2 fs points', y='s2 fs momentum', hue='s1 win', data=less_data)
plt.show()

sns.relplot(x='s1 fs points', y='s2 fs points', hue='comeback', data=data)
plt.show()

sns.relplot(x='s1 fs momentum', y='s2 fs momentum', hue='comeback', data=data)
plt.show()
