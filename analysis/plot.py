import pandas as pd
import seaborn as sns
from seaborn import FacetGrid
import matplotlib.pyplot as plt
import re
import numpy as np

D = pd.read_csv('/home/kabird/STNDT_fewshot/results3.csv',index_col=0)

D.drop(columns=['index'],inplace=True)
print(D)
# D['path'] = D['path'].str.split('/').str[3].str.split('_CHECK').str[0]
print(len(D['path'].unique()))
score_vars = [c for c in D.columns if 'co-bps' in c ] #+ ['vel R2','psth R2','fp-bps'] #and 'shot' in c


# Dscores = D.melt(id_vars=['fewshot_learner','path'],value_vars=score_vars,value_name='score',var_name='score_type')
# selectDscores = Dscores[Dscores.path.isin(Dscores.path.unique()[:2])]
# print(selectDscores.columns)


# Dscores = D[D.path.isin(D.path.unique()[:2])][score_vars].T
# Dscores.columns = D.path.unique()[:2]

# print(Dscores)

# fig,ax = plt.subplots()
# sns.scatterplot(
#     x=D.path.unique()[0],
#     y=D.path.unique()[1],
#     data=Dscores,
#     ax=ax
# )
# xlims,ylims = ax.get_xlim(),ax.get_ylim()
# ax.plot([-1,1],[-1,1],ls='dashed',c='black')
# ax.set_xlim(xlims)
# ax.set_ylim(xlims)
# ax.set_aspect('equal')
# fig.savefig('plots/two_models.png')

print(D.columns)
Dbest = D[D['co-bps']>D['co-bps'].max()-2e-2]
Dscores = Dbest[Dbest.path.isin(Dbest.path.unique()[:])][score_vars].T
Dscores.columns = Dbest.path.unique()[:]

# # Create a PairGrid
# grid = sns.PairGrid(Dscores)

# # Map scatter plots to the grid
# grid.map_lower(sns.scatterplot, s=10)
# # grid.map_lower(sns.kdeplot, colors='C0')
# # grid.map_diag(sns.histplot, kde_kws={'color': 'C0'})

# xlim = (0,0.5)
# ylim = (0,0.5)

# # Add a dashed black line for x=y
# for ax in grid.axes.flat:
#     ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".3")
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
#     # ax.set_aspect('equal')

# # Set font size for axis labels
# for ax in grid.axes[-1,:]:
#     ax.set_xlabel('')
# for ax in grid.axes[:,0]:
#     ax.set_ylabel('')

# # Optional: Add more customization if needed
# grid.fig.suptitle('Pairwise Scatter Plots')

# # Save the figure to a PNG file
# plt.savefig('plots/pairwise_scatter_plots.png',dpi=200)


fewshot_vars = [c for c in D.columns if 'co-bps' in c and 'shot' in c ]

df = D[fewshot_vars]
### computing averages
pattern = r'(\d+)shot_id\d+ co-bps'
ks = set()
for column in df.columns:
    match = re.match(pattern, column)
    if match:
        ks.add(int(match.group(1)))

print(ks)
# Calculate mean and std for each K
result = {}
for k in ks:
    pattern_k = re.compile(f'{k}shot_id(\d+) co-bps')
    columns = [column for column in df.columns if pattern_k.match(column)]
    subset = df[columns]
    mean_column_name = f'mean{k}shot co-bps'
    std_column_name = f'std{k}shot co-bps'
    result[mean_column_name] = subset.mean(axis=1)
    result[std_column_name] = subset.std(axis=1)/np.sqrt(len(columns))

# Create a new DataFrame
result_df = pd.DataFrame(result)
print(result_df)
D = pd.concat([D,result_df],axis=1)

k = 16

fig,ax = plt.subplots()

x = D['co-bps']
y = D[f'mean{k}shot co-bps']
errors = D[f'std{k}shot co-bps']
ax.errorbar(x, y, yerr=errors, fmt='.', markersize=7, color='black', capsize=2)
# sns.scatterplot(x='co-bps',y='mean16shot co-bps',ax=ax,data=D)
ax.set_xlabel('co-bps')
ax.set_ylabel(f'mean{k}shot co-bps')
sns.despine(fig,ax=ax)

fig.savefig(f'plots/mean{k}shot_vs_co-bps.png',dpi=300)
plt.close(fig)

fig,ax = plt.subplots()
x = D['co-bps']
y = D['vel R2']
# errors = D['std16shot co-bps']
# ax.errorbar(x, y, yerr=errors, fmt='.', markersize=7, color='black', capsize=2)
sns.scatterplot(x='co-bps',y='vel R2',ax=ax,data=D)
ax.set_xlabel('co-bps')
ax.set_ylabel('vel R2')
sns.despine(fig,ax=ax)

fig.savefig('plots/velR2_vs_co-bps.png',dpi=300)
plt.close(fig)

fig,ax = plt.subplots()
x = D['vel R2']
y = D[f'mean{k}shot co-bps']
errors = D[f'std{k}shot co-bps']
ax.errorbar(x, y, yerr=errors, fmt='.', markersize=7, color='black', capsize=2)
# sns.scatterplot(x='co-bps',y='mean16shot co-bps',ax=ax,data=D)
ax.set_xlabel('vel R2')
ax.set_ylabel(f'mean{k}shot co-bps')
sns.despine(fig,ax=ax)

fig.savefig(f'plots/mean{k}shot_vs_velR2.png',dpi=300)
plt.close(fig)

fig,ax = plt.subplots()
x = D['co-bps']
y = D[f'mean{k}shot co-bps']
errors = D[f'std{k}shot co-bps']
ax.errorbar(x, y, yerr=errors, fmt='.', markersize=7, color='black', capsize=2)
# sns.scatterplot(x='co-bps',y='mean16shot co-bps',ax=ax,data=D)
ax.set_xlabel('co-bps')
ax.set_ylabel(f'mean{k}shot co-bps')
sns.despine(fig,ax=ax)

fig.savefig(f'plots/mean{k}shot_vs_co-bps.png',dpi=300)
plt.close(fig)

fig,ax = plt.subplots()
x = D['psth R2']
y = D[f'mean{k}shot co-bps']
errors = D[f'std{k}shot co-bps']
ax.errorbar(x, y, yerr=errors, fmt='.', markersize=7, color='black', capsize=2)
# sns.scatterplot(x='co-bps',y='mean16shot co-bps',ax=ax,data=D)
ax.set_xlabel('psth R2')
ax.set_ylabel(f'mean{k}shot co-bps')
sns.despine(fig,ax=ax)

fig.savefig(f'plots/mean{k}shot_vs_psthR2.png',dpi=300)
plt.close(fig)

fig,ax = plt.subplots()
# x = D['psth R2']
# y = D['co-bps']
# errors = D[f'co-bps']
# ax.errorbar(x, y, yerr=errors, fmt='.', markersize=7, color='black', capsize=2)
sns.scatterplot(x='co-bps',y='psth R2',ax=ax,data=D)
ax.set_ylabel('psth R2')
ax.set_xlabel(f'co-bps')
sns.despine(fig,ax=ax)

fig.savefig(f'plots/co-bps_vs_psthR2.png',dpi=300)
plt.close(fig)