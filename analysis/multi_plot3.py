import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np

from scipy.stats import pearsonr, spearmanr
from plot_cross_decoding import compute_colsums, plot_cross_decoding_scores_from_dataframe

# Define the result files to be processed
result_files = [
    # {
    #     'method' : 'lfads-torch',
    #     'path' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240519_194718_MultiFewshot/concatenated_results.csv', 
    #     'co-bps threshold'  : 5e-3,
    # },
    {
        'method'  : 'STNDT',
        'path' : '/home/kabird/STNDT_fewshot/ray_results/mc_maze_20_lite_all9.csv', # STNDT mc_maze
        'co-bps threshold'  : 5e-2,
    },
]

# Read and combine the result files
D = []
for f in result_files:
    d_ = pd.read_csv(f['path'], index_col=0)
    d_['method'] = f['method']
    D.append(d_)

D = pd.concat(D, axis=0)
D.drop(columns=['index'], inplace=True)
# D.index = D['path'].str.split('/').str[7].str.split('_').str[3].astype(int)
D.index = D['id']

# Check if the cross decoding scores file exists
cross_decoding_scores_path = result_files[0]['path'].replace('.csv', '_cross_decoding_scores.csv')
if os.path.exists(cross_decoding_scores_path):
    scores = pd.read_csv(cross_decoding_scores_path)
    scores['from_id'] = scores['from_id'].str.split('/').str[7].str.split('_').str[3].astype(int)
    scores['to_id'] = scores['to_id'].str.split('/').str[7].str.split('_').str[3].astype(int)
    scores.drop(columns=['from_to_index', 'from', 'to'], inplace=True)

    square_score_dataframe = scores.pivot(
        index='from_id',
        columns='to_id',
        values='score'
    )

    idx_intersection = square_score_dataframe.index.intersection(D.index)
    D = D.loc[idx_intersection]
    D['1-R2 column sum new'] = (1 - square_score_dataframe.loc[:, idx_intersection].values).mean(axis=0)
    D['1-R2 row sum new'] = (1 - square_score_dataframe.loc[idx_intersection, :].values).mean(axis=1)
else:
    print(f"Cross decoding scores file not found: {cross_decoding_scores_path}")
    scores = None

# Compute averages for each k-shot value and method
pattern = r'(\d+)shot_id\d+ co-bps:(.*)'
ks = {(int(re.match(pattern, column).group(1)), re.match(pattern, column).group(2)) 
      for column in D.columns if re.match(pattern, column)}

method_names = [
    # 'sklearn_glm.fit_poisson_parallel(alpha=0.1,max_iter=500)',
    'sklearn_glm.fit_poisson_parallel(alpha=0.01,max_iter=500)',
    'sklearn_glm.fit_poisson_parallel(alpha=0.001,max_iter=500)',
    'sklearn_glm.fit_poisson_parallel(alpha=0.0001,max_iter=500)'
]

result = {}
for k_val, method_name in ks:
    columns = [column for column in D.columns if re.match(f'{k_val}shot_id\d+ co-bps:{re.escape(method_name)}', column)]
    subset = D[columns]
    mean_column_name = f'mean{k_val}shot co-bps:{method_name}'
    std_column_name = f'std{k_val}shot co-bps:{method_name}'
    result[mean_column_name] = subset.mean(axis=1)
    result[std_column_name] = subset.std(axis=1)

result_df = pd.DataFrame(result)
D = pd.concat([D, result_df], axis=1)

D.to_csv(result_files[0]['path'].replace('.csv', '_with_means.csv'))

# Clean column names
D.columns = [c.replace('co_bps', 'co-bps') for c in D.columns]
D.columns = [c.replace(':', '\n') for c in D.columns]

# Filter data for plotting
D = D[D['co-bps'] > 0]

# Determine the columns to plot
if scores is not None:
    to_plot = [
        f'mean{k}shot co-bps\n{method_name}' for k in [128, 256, 1024] for method_name in method_names
    ] + ['1-R2 column sum new', '1-R2 row sum new', 'co-bps', 'method']
else:
    to_plot = [
        f'mean{k}shot co-bps\n{method_name}' for k in [128, 256, 1024] for method_name in method_names
    ] + ['co-bps', 'method']

# Prepare and plot data
dataset_name = D.dataset.unique()[0]
select_Dscores = D[D['dataset'] == dataset_name]
select_Dscores = select_Dscores[select_Dscores['co-bps'] > select_Dscores['co-bps'].max() - result_files[0]['co-bps threshold']]

# Plot heatmap of cross-correlation matrix
df = select_Dscores[to_plot]
corr_matrix = df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax, annot_kws={'size': 7})
ax.set_title('Heatmap of Cross-Correlation Matrix')
ax.tick_params(axis='both', which='major', labelsize=5)
fig.tight_layout()
fig.savefig(f'plots/NLBmetrics_correlations_{result_files[0]["method"]}_{dataset_name}_verybest_only_newcolsum_only.png', dpi=300)


################## Pair grid

grid = sns.PairGrid(data=select_Dscores[to_plot], hue='method')
# grid = sns.pairplot(data = select_Dscores, hue='fewshot_method', vars=['value','co-bps', 'vel R2', 'psth R2', 'fp-bps'],diag_kind=None)
grid.map_upper(sns.scatterplot, s=10)
grid.map_lower(sns.scatterplot, s=10)
for ax in grid.axes[:,0]:
    ax.yaxis.label.set_size(5)
for ax in grid.axes[-1,:]:
    ax.xaxis.label.set_size(5)

g = grid
# Loop over the axes in the Pairplot
for i, row_axes in enumerate(g.axes):
    for j, ax in enumerate(row_axes):
        xlabel = g.x_vars[j]
        ylabel = g.y_vars[i]
        if i != j:  # Avoid diagonal plots
            data1 = select_Dscores[grid.x_vars[j]]
            data2 = select_Dscores[grid.y_vars[i]]
            # corr_coef = np.corrcoef(data1, data2)[0, 1]
            # ax.set_title(f'Correlation: {corr_coef:.2f}')
            ax.set_title(f'Spearman:{spearmanr(data1, data2)[0]:.2f}')
            
        if 'bps' in xlabel and 'bps' in ylabel and i != j:
            # Calculate limits based on the max of both axes for square aspect ratio
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            max_limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
            min_limit = min(ax.get_xlim()[0], ax.get_ylim()[0])
            ax.plot([min_limit, max_limit], [min_limit, max_limit], ls="--", c="black")
            # ax.set_xlim(*[min_limit, max_limit])
            # ax.set_ylim(*[min_limit, max_limit])
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

# for ax in grid.axes[0,:]:
#     ax.set_ylim(0)
# for ax in grid.axes[:,0]:
#     ax.set_xlim(0)
# plt.legend(title='Method', loc='upper right', bbox_to_anchor=(1.25, 1))

handles = g.axes[0, -1].get_legend_handles_labels()[0]
labels = g.axes[0, -1].get_legend_handles_labels()[1]
# g.axes[0,-1].legend()

g.fig.legend(handles, labels, loc='upper right', title='Method') #
fig = g.fig
fig.suptitle('NLB metrics '+ dataset_name)
fig.tight_layout()
# fig.savefig(f'plots/NLBmetrics_{dataset_name}_best_only.png',dpi=200)
fig.savefig(f'plots/NLBmetrics_{result_files[0]["method"]}_{dataset_name}_rates-as-latents_verybest_only_newcolsum_only.png',dpi=200)

# If cross decoding scores are available, create scatter plots
if scores is not None:
    kshot_title = f'mean{256}shot co-bps\n{method_names[2]}'
    kshot_title_stderr = kshot_title.replace('mean', 'std')

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for j, score_title in enumerate([kshot_title, 'co-bps']):
        ax = axs[0, j]
        sns.scatterplot(data=select_Dscores, x='1-R2 column sum new', y=score_title, ax=ax)
        if score_title == kshot_title:
            ax.errorbar(select_Dscores['1-R2 column sum new'], select_Dscores[score_title], yerr=select_Dscores[kshot_title_stderr], fmt='o')

        for i in select_Dscores.index:
            ax.text(select_Dscores.loc[i]['1-R2 column sum new'], select_Dscores.loc[i][score_title], str(i), color='red', ha='right')

        ax = axs[1, j]
        sns.scatterplot(data=select_Dscores, x='1-R2 row sum new', y=score_title, ax=ax)
        if score_title == kshot_title:
            ax.errorbar(select_Dscores['1-R2 row sum new'], select_Dscores[kshot_title], yerr=select_Dscores[kshot_title_stderr], fmt='o')

        for i in select_Dscores.index:
            ax.text(select_Dscores.loc[i]['1-R2 row sum new'], select_Dscores.loc[i][score_title], str(i), color='red', ha='right')

    fig.tight_layout()
    fig.savefig(f'plots/{result_files[0]["method"]}_{256}shot{method_names[2]}_and_rowcolsum_indices.png', dpi=300)

