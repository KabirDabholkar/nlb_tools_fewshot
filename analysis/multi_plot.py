import pandas as pd
import seaborn as sns
from seaborn import FacetGrid
import matplotlib.pyplot as plt
import re
import numpy as np

result_files = [
    # {
    #     'method'  : 'STNDT',
    #     'path' : '/home/kabird/STNDT_fewshot/results5.csv', # STNDT mc_maze
    # },
    {
        'method'  : 'STNDT',
        'path' : '/home/kabird/STNDT_fewshot/ray_results/mc_maze_20_lite_all4.csv', # STNDT mc_maze
    },
    # {
    #     'method' : 'lfads-torch',
    #     'path' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_dmfc_rsg/240501_030533_MultiFewshot/concatenated_results.csv', # LFADS dmfc_rsg
    # }
    
]

D = []
for f in result_files:
    d_ = pd.read_csv(f['path'],index_col=0)
    d_['method'] = f['method']
    # print(d_.head(2).iloc[:,:5])   
    D.append(d_)

D = pd.concat(D,axis=0)
print(D.columns)
# print(D.groupby('method').head(2))

print(D.size)
score_vars = [c for c in D.columns if 'co-bps' in c ] + ['vel R2','psth R2','fp-bps'] #and 'shot' in c

fewshot_vars = [c for c in D.columns if 'co-bps' in c and 'shot' in c ]

###### computing averages for each k #########
df = D[fewshot_vars]
# pattern = r'(\d+)shot_id\d+ co-bps'
pattern = r'(\d+)shot_id\d+ co-bps:(.*)'
ks = set()
for column in df.columns:
    match = re.match(pattern, column)
    if match:
        ks.add((int(match.group(1)), match.group(2)))
        # ks.add(int(match.group(1)))

print('k values',ks)
# Calculate mean and std for each K
# result = {}
# for k in ks:
#     pattern_k = re.compile(f'{k}shot_id(\d+) co-bps')
#     columns = [column for column in df.columns if pattern_k.match(column)]
#     subset = df[columns]
#     mean_column_name = f'mean{k}shot co-bps'
#     std_column_name = f'std{k}shot co-bps'
#     result[mean_column_name] = subset.mean(axis=1)
#     result[std_column_name] = subset.std(axis=1)/np.sqrt(len(columns))
# Calculate mean and std for each K

# method_name1 = 'torch_glm.fit_poisson'
# method_name2 = 'sklearn_glm.fit_poisson_parallel'
method_name1 = 'sklearn_glm.fit_poisson_parallel(alpha=0.0,max_iter=500)'
method_name2 = 'sklearn_glm.fit_poisson_parallel(alpha=0.1,max_iter=500)'

result = {}
for k in ks:
    k_val, method_name = k
    # print(k_val,method_name)
    # print(df.columns)
    # print(f'{k_val}shot_id\d+ co-bps:{method_name}')
    # print(re.match(f'{k_val}shot_id\d+ co-bps:', '512shot_id0 co-bps:sklearn_glm.fit_poisson_parallel(alpha=0.0,max_iter=500)'))
    # print(re.match(f'{k_val}shot_id\d+ co-bps:', '512shot_id0 co-bps:sklearn_glm.fit_poisson_parallel(alpha=0.0,max_iter=1000)'))
    # print(re.match(f'{k_val}shot_id\d+ co-bps:'+re.escape(mea), '512shot_id0 co-bps:sklearn_glm.fit_poisson_parallel(alpha=0.0,max_iter=1000)'))
    # print(re.match(f'{k_val}shot_id\d+ co-bps:'+re.escape(method_name), '512shot_id0 co-bps:'))
    columns = [column for column in df.columns if re.match(f'{k_val}shot_id\d+ co-bps:'+re.escape(method_name), column)]
    print('columns',columns)
    subset = df[columns]
    print('subset',subset)
    mean_column_name = f'mean{k_val}shot co-bps:{method_name}'
    std_column_name = f'std{k_val}shot co-bps:{method_name}'
    result[mean_column_name] = subset.mean(axis=1)
    result[std_column_name] = subset.std(axis=1)

# concat statististics
result_df = pd.DataFrame(result)
# print(result_df)
D = pd.concat([D,result_df],axis=1)
# print(D.head(2))
###########################
# print(D.dataset.unique())

D = D[D['co-bps']>0]
# D.dropna(subset=[f'mean{32}shot co-bps:{method_name}'],inplace=True)
# D = D[D[f'mean{32}shot co-bps:sklearn_glm.fit_poisson_parallel']>-1]
# Dbest = D[D['co-bps']>D['co-bps'].max()-2e-2]
# print('max',D[f'mean{32}shot co-bps:{method_name}'].max())
# print('any nans',np.isnan(D[f'mean{32}shot co-bps:{method_name}'].values).any())
print(D.dataset.unique())
Dscores = D #Dbest[Dbest.path.isin(Dbest.path.unique()[:])]#.T
# Dscores.columns = Dbest.path.unique()[:]

def prepare_dataframe_for_plotting(df):
    """
    Reshape the DataFrame for Seaborn pair plotting.

    Parameters:
    df (DataFrame): The original DataFrame with complex column names.
    
    Returns:
    DataFrame: A reshaped DataFrame with separate 'shot', 'fewshot_method', and 'value' columns.
    """
    # Melt the DataFrame to long format
    df_long = df.melt(id_vars=['co-bps', 'vel R2', 'psth R2', 'fp-bps'], var_name='variable', value_name='value')
    
    # Extract shot size and method from the column names
    df_long['shot'] = df_long['variable'].str.extract('(\d+)shot')
    df_long['fewshot_method'] = df_long['variable'].apply(lambda x: x.split(':')[-1])
    
    # Drop the original 'variable' column as it's no longer needed
    df_long.drop(columns=['variable'], inplace=True)
    
    return df_long

to_plot = ([f'mean{k}shot co-bps:{method_name1}' for k in [512]]
           +[f'mean{k}shot co-bps:{method_name2}' for k in [512]] 
           + ['co-bps'] + ['vel R2','psth R2','fp-bps'] )# + ['method'])

print(Dscores)

dataset_name = Dscores.dataset.unique()[0] #'mc_maze_20_split'
print(Dscores.dataset.unique())
for dataset_name in Dscores.dataset.unique()[:1]:
    select_Dscores = Dscores[Dscores.dataset==dataset_name][to_plot]
    # select_Dscores = prepare_dataframe_for_plotting(select_Dscores)
    # print(select_Dscores.tail(4))


    grid = sns.PairGrid(data=select_Dscores)
    # grid = sns.pairplot(data = select_Dscores, hue='fewshot_method', vars=['value','co-bps', 'vel R2', 'psth R2', 'fp-bps'],diag_kind=None)
    grid.map_upper(sns.scatterplot, s=10)
    for ax in grid.axes[:,0]:
        ax.yaxis.label.set_size(7)
    for ax in grid.axes[-1,:]:
        ax.xaxis.label.set_size(7)
    # for ax in grid.axes[0,:]:
    #     ax.set_ylim(0)
    # for ax in grid.axes[:,0]:
    #     ax.set_xlim(0)
    # plt.legend(title='Method', loc='upper right', bbox_to_anchor=(1.25, 1))
    fig = grid.fig
    fig.suptitle('NLB metrics '+ dataset_name)
    fig.tight_layout()
    fig.savefig(f'plots/NLBmetrics_{dataset_name}.png',dpi=200)

#######################




# melted_D = transform_dataframe(D[fewshot_vars])
# print(melted_D.head(4))

# print(melted_D)


#############################

for dataset_name in Dscores.dataset.unique()[:1]:
    select_Dscores = Dscores[Dscores.dataset==dataset_name]
    # print(select_Dscores[['512shot_id0 co-bps:sklearn_glm.fit_poisson_parallel','512shot_id0 co-bps:torch_glm.fit_poisson']].head(20))
    # print(select_Dscores[['sklearn_glm.fit_poisson_parallel(alpha=0.0,max_iter=500)']]) 
    # print(select_Dscores.columns)
    fig, ax = plt.subplots()
    sns.scatterplot(
        # x='512shot_id0 co-bps:sklearn_glm.fit_poisson_parallel',
        # y='512shot_id0 co-bps:torch_glm.fit_poisson',
        x='512shot_id0 co-bps:sklearn_glm.fit_poisson_parallel(alpha=0.0,max_iter=500)',
        y='512shot_id0 co-bps:sklearn_glm.fit_poisson_parallel(alpha=0.1,max_iter=500)',
        data=select_Dscores,
        ax=ax
    )
    ax.plot([0,.3],[0,.3],ls='dashed',c='black')
    fig.savefig(f'plots/fewshotmethod_comparison_{dataset_name}.png',dpi=200)