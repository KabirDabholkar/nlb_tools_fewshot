import pandas as pd
import seaborn as sns
from seaborn import FacetGrid
import matplotlib.pyplot as plt
import re
import numpy as np

result_files = [
    {
        'method'  : 'STNDT',
        'path' : '/home/kabird/STNDT_fewshot/results5.csv', # STNDT mc_maze
    },
    {
        'method' : 'lfads-torch',
        'path' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_dmfc_rsg/240501_030533_MultiFewshot/concatenated_results.csv', # LFADS dmfc_rsg
    }
    
]

D = []
for f in result_files:
    d_ = pd.read_csv(f['path'],index_col=0)
    d_['method'] = f['method']
    # print(d_.head(2).iloc[:,:5])   
    D.append(d_)

D = pd.concat(D,axis=0)

# print(D.groupby('method').head(2))

print(D.size)
score_vars = [c for c in D.columns if 'co-bps' in c ] + ['vel R2','psth R2','fp-bps'] #and 'shot' in c

fewshot_vars = [c for c in D.columns if 'co-bps' in c and 'shot' in c ]

###### computing averages for each k #########
df = D[fewshot_vars]
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

# concat statististics
result_df = pd.DataFrame(result)
# print(result_df)
D = pd.concat([D,result_df],axis=1)
###########################
print(D.dataset.unique())

D = D[D['co-bps']>0]
# Dbest = D[D['co-bps']>D['co-bps'].max()-2e-2]
print(D.dataset.unique())
Dscores = D #Dbest[Dbest.path.isin(Dbest.path.unique()[:])]#.T
# Dscores.columns = Dbest.path.unique()[:]


to_plot = [f'mean{k}shot co-bps' for k in [16,1024]] + ['co-bps'] + ['vel R2','psth R2','fp-bps'] + ['method']

# dataset_name = Dscores.dataset.unique()[0] #'mc_maze_20_split'
print(Dscores.dataset.unique())
for dataset_name in Dscores.dataset.unique()[:1]:
    select_Dscores = Dscores[Dscores.dataset==dataset_name][to_plot]
    print(select_Dscores)


    grid = sns.PairGrid(data=select_Dscores)
    grid.map_upper(sns.scatterplot, s=10)
    fig = grid.fig
    fig.suptitle('NLB metrics '+ dataset_name)
    fig.tight_layout()
    fig.savefig(f'plots/NLBmetrics_{dataset_name}.png',dpi=200)

