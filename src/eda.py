import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

data = pd.read_csv(f'data{os.path.sep}wrangled_data.csv', index_col=0)

# Data visualization
plot_data = data[['p1_win_fs', 'p1_win']]

plot_data.groupby('p1_win_fs').p1_win.value_counts().unstack(0).plot.barh()

# %%
plot = sns.catplot(x='p1_win', y='fs_s1_points', kind='box', data=data)
plt.show()
# %%
sns.catplot(x='p1_win', y='fs_s2_points', kind='box', data=data)