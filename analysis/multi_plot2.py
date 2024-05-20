import pandas as pd
import seaborn as sns
from seaborn import FacetGrid
import matplotlib.pyplot as plt
import re
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr
from plot_cross_decoding import compute_colsums, plot_cross_decoding_scores_from_dataframe

result_files = [
    # {
    #     'method'  : 'STNDT',
    #     'path' : '/home/kabird/STNDT_fewshot/results5.csv', # STNDT mc_maze
    # },
    # {
    #     'method'  : 'STNDT',
    #     'path' : '/home/kabird/STNDT_fewshot/ray_results/mc_maze_20_lite_all7.csv', # STNDT mc_maze
    # },
    # {
    #     'method' : 'lfads-torch',
    #     'path' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_dmfc_rsg/240501_030533_MultiFewshot/concatenated_results.csv', # LFADS dmfc_rsg
    # }
    # {
    #     'method' : 'lfads-torch',
    #     'path' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240511_141307_MultiFewshot/concatenated_results.csv', # LFADS mc_maze_20 22 really heldout
    # }
    # {
    #     'method' : 'lfads-torch',
    #     'path' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240514_000059_MultiFewshot/concatenated_results.csv', # LFADS mc_maze_20 22 really heldout MAIN
    # },
    # {
    #     'method' : 'lfads-torch-rates-pred',
    #     'path' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240518_134433_MultiFewshot/concatenated_with_kshot_and_path_to_latents.csv', # LFADS mc_maze_20 22 really heldout
    # },
    # {
    #     'method' : 'lfads-torch',
    #     'path' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240519_184438_MultiFewshot/concatenated_results.csv', # LFADS mc_maze_20 22 really heldout
    # },
    {
        'method' : 'lfads-torch',
        'path' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240519_194718_MultiFewshot/concatenated_results.csv', # LFADS mc_maze_20 22 really heldout
    },
]


D = []
for f in result_files:
    d_ = pd.read_csv(f['path'],index_col=0)
    d_['method'] = f['method']
    # print(d_.head(2).iloc[:,:5])   
    D.append(d_)

D = pd.concat(D,axis=0)
D.drop(columns=['index'],inplace=True)
# D.dropna(subset=['path'],inplace=True)
# print(D['path'].str.split('/').str[7].unique())
D.index = D['path'].str.split('/').str[7].str.split('_').str[3].astype(int)
# print(D.columns)
# print(D.groupby('method').head(2))



# ############

scores = pd.read_csv(result_files[0]['path'].replace('.csv','_cross_decoding_scores.csv'))

scores['from_id'] = scores['from_id'].str.split('/').str[7].str.split('_').str[3].astype(int)
scores['to_id'] = scores['to_id'].str.split('/').str[7].str.split('_').str[3].astype(int)
scores.drop(columns=['from_to_index','from','to'],inplace=True)

# # scores_old_and_new = pd.merge(scores, scores_old, on=['from_id', 'to_id'],suffixes=('_new','_old'))


# print(scores.shape)

square_score_dataframe = scores.pivot(
    index = 'from_id',
    columns = 'to_id',
    values = 'score'
)

# print(square_score_dataframe.index,len(square_score_dataframe.index))

idx_intersection = square_score_dataframe.index.intersection(D.index)
print('idx_intersection',idx_intersection)
D = D.loc[idx_intersection]
# D['1-R2 column sum new'] = (1-square_score_dataframe.values).mean(axis=0)
# D['1-R2 row sum new'] = (1-square_score_dataframe.values).mean(axis=1)
D['1-R2 column sum new'] = (1-square_score_dataframe.loc[:,idx_intersection].values).mean(axis=0)
D['1-R2 row sum new'] = (1-square_score_dataframe.loc[idx_intersection,:].values).mean(axis=1)


print(D.iloc[:4,:4])

# D = compute_colsums(
#     D,#[D['co-bps']>D['co-bps'].max()-1e-2],
#     scores
# )
# D.rename(columns={'1-R^2 column sum': '1-R2 column sum new'}, inplace=True)
# D = compute_colsums(
#     D,#[D['co-bps']>D['co-bps'].max()-1e-2],
#     scores,
#     axis=1
# )
# D.rename(columns={'1-R^2 column sum': '1-R2 row sum new'}, inplace=True)


# # #############


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


# method_name1 = 'torch_glm.fit_poisson'
# method_name2 = 'sklearn_glm.fit_poisson_parallel'
method_name1 = 'sklearn_glm.fit_poisson_parallel(alpha=0.1,max_iter=500)'
method_name2 = 'sklearn_glm.fit_poisson_parallel(alpha=0.01,max_iter=500)'
method_name3 = 'sklearn_glm.fit_poisson_parallel(alpha=0.001,max_iter=500)'
# method_name4 = 'torch_glm.fit_poisson'


result = {}
for k in ks:
    k_val, method_name = k

    columns = [column for column in df.columns if re.match(f'{k_val}shot_id\d+ co-bps:'+re.escape(method_name), column)]
    # print('columns',columns)
    subset = df[columns]
    # print('subset',subset)
    mean_column_name = f'mean{k_val}shot co-bps:{method_name}'
    std_column_name = f'std{k_val}shot co-bps:{method_name}'
    result[mean_column_name] = subset.mean(axis=1)
    result[std_column_name] = subset.std(axis=1)
    print(std_column_name)

# concat statististics
result_df = pd.DataFrame(result)
# print(result_df)
D = pd.concat([D,result_df],axis=1)

D.to_csv(result_files[0]['path'].replace('.csv','_with_means.csv'))

###############

D.columns  =  [c.replace('co_bps','co-bps') for c in  D.columns]


# print(D.head(2))
###########################
# print(D.dataset.unique())

D = D[D['co-bps']>0]

print(D.dataset.unique())
Dscores = D #Dbest[Dbest.path.isin(Dbest.path.unique()[:])]#.T


D.columns = [c.replace(':','\n') for c in D.columns]


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

to_plot = ([]# [f'mean{k}shot co-bps\n{method_name1}' for k in [32,64,1024]]
            # + [f'mean{k}shot co-bps\n{method_name2}' for k in [32,64,1024]] 
            # + [f'mean{k}shot co-bps\n{method_name3}' for k in [32,64,1024]] 
            # + [f'mean{k}shot co-bps\n{method_name4}' for k in [64,1024]] 
            # +[f'valid/{k}shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_recon_bps'for k in [100,1000]]
            # +[f'valid/{k}shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_reallyheldout_bps' for k in [100,1000]]
            # +['1-R2 column sum old']
            + [f'mean{k}shot co-bps\n{method_name1}' for k in [128,256,1024]]
            + [f'mean{k}shot co-bps\n{method_name2}' for k in [128,256,1024]]
            + [f'mean{k}shot co-bps\n{method_name3}' for k in [128,256,1024]]
            + ['1-R2 column sum new']
            + ['1-R2 row sum new']
           + ['co-bps'] 
        #    + ['valid/co-bps']
        #    + ['vel R2','psth R2','fp-bps'] 
           + ['method'])

print('Method unique:',Dscores['method'].unique())

dataset_name = Dscores.dataset.unique()[0] #'mc_maze_20_split'
print(Dscores.dataset.unique())
for dataset_name in Dscores.dataset.unique()[:1]:
    select_Dscores = Dscores[Dscores.dataset==dataset_name]
    
    # select_Dscores = select_Dscores[select_Dscores[f'mean{1024}shot co-bps\n{method_name3}'] > select_Dscores[f'mean{1024}shot co-bps\n{method_name3}'].max()-1e-2]
    select_Dscores = select_Dscores[select_Dscores['co-bps']>select_Dscores['co-bps'].max()-0.5e-2]
    # print('median co-bps index',select_Dscores['co-bps'].sort_values(ascending=False).head(0).index)
    print('max co-bps index',select_Dscores['co-bps'].sort_values(ascending=False).head(1).index)
    # select_Dscores = prepare_dataframe_for_plotting(select_Dscores)
    # print(select_Dscores.tail(4))



    ### cross correlation matrix
    # Calculate the correlation matrix
    df = select_Dscores[to_plot]
    corr_matrix = df.corr()

    # Plot the heatmap
    
    fig,ax= plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,ax=ax, annot_kws={'size':7}) #, 
    ax.set_title('Heatmap of Cross-Correlation Matrix')
    ax.tick_params(axis='both', which='major', labelsize=5)
    fig.tight_layout()
    fig.savefig(f'plots/NLBmetrics_correlations_{dataset_name}_verybest_only_newcolsum_only.png',dpi=300)

    print('saved correlations plot')

    kshot_title = f'mean{256}shot co-bps\n{method_name3}'
    kshot_title_stderr = kshot_title.replace('mean','std')

    # ###### cross decoding
    # best_k_shot_idx = select_Dscores.sort_values(
    #         by=kshot_title,
    #         ascending=False).head(5).index
    
    # worst_k_shot_idx = select_Dscores.sort_values(
    #         by=kshot_title,
    #         ascending=True).head(5).index
    # print('worst',worst_k_shot_idx)
    # print('best',best_k_shot_idx)

    

    # # Create a scatter plot
    data = select_Dscores

    fig,axs = plt.subplots(2,2,figsize=(12,10))

    for j,score_title in enumerate([kshot_title,'co-bps']):
        ax = axs[0,j]
        sns.scatterplot(data=data, x='1-R2 column sum new', y=score_title,ax=ax)
        if score_title==kshot_title:
            ax.errorbar(data['1-R2 column sum new'],data[score_title],yerr=data[kshot_title_stderr],fmt='o')
        ax.set_title

        # Annotate each point with its DataFrame index
        for i in data.index:
            ax.text(data.loc[i]['1-R2 column sum new'], data.loc[i][score_title], str(i), color='red', ha='right')

        ax = axs[1,j]
        sns.scatterplot(data=data, x='1-R2 row sum new', y=score_title,ax=ax)
        if score_title==kshot_title:
            ax.errorbar(data['1-R2 row sum new'],data[kshot_title],yerr=data[kshot_title_stderr],fmt='o')

        # Annotate each point with its DataFrame index
        for i in data.index:
            ax.text(data.loc[i]['1-R2 row sum new'], data.loc[i][score_title], str(i), color='red', ha='right')
    fig.tight_layout()
    fig.savefig(f'plots/{256}shot{method_name3}_and_rowcolsum_indices.png',dpi=300)

    # fig,axs = plt.subplots(2,1,figsize=(6,10))

    # ax = axs[0]
    # sns.scatterplot(data=data, x='1-R2 column sum new', y=kshot_title,ax=ax)
    # ax.errorbar(data['1-R2 column sum new'],data[kshot_title],yerr=data[kshot_title_stderr],fmt='o')

    # # Annotate each point with its DataFrame index
    # for i in data.index:
    #     ax.text(data.loc[i]['1-R2 column sum new'], data.loc[i][kshot_title], str(i), color='red', ha='right')

    # ax = axs[1]
    # sns.scatterplot(data=data, x='1-R2 row sum new', y=kshot_title,ax=ax)
    # ax.errorbar(data['1-R2 row sum new'],data[kshot_title],yerr=data[kshot_title_stderr],fmt='o')

    # # Annotate each point with its DataFrame index
    # for i in data.index:
    #     ax.text(data.loc[i]['1-R2 row sum new'], data.loc[i][kshot_title], str(i), color='red', ha='right')
    # fig.tight_layout()
    # fig.savefig(f'plots/{128}shot{method_name3}_and_rowcolsum_indices.png',dpi=300)

    
    # plot_cross_decoding_scores_from_dataframe(
    #     score_dataframe = scores,
    #     select_indices = select_Dscores.index,
    #     # highlight_cols=best_k_shot_idx,
    #     # highlight_rows=best_k_shot_idx,
    #     highlight_cols=best_k_shot_idx,
    #     highlight_rows=best_k_shot_idx,
    #     figsavepath='plots/cross_decoding_with_highlights.png',
    #     style='triangle'
    # )

    # select_intersect_idx = scores.index.intersection(select_Dscores.index)
    # score_dataframe = scores
    # square_score_dataframe = score_dataframe.pivot(
    #     index = 'from_id',
    #     columns = 'to_id',
    #     values = 'score'
    # )
    # square_score_dataframe = square_score_dataframe.loc[select_intersect_idx,select_intersect_idx]
    # square_score_dataframe.to_csv('plots/LFADS_mc_maze_20_cross_decoding_matrix.csv')
    # square_score_dataframe_loaded = pd.read_csv('plots/LFADS_mc_maze_20_cross_decoding_matrix.csv',index_col=0)
    # print('max col mean',(1 - square_score_dataframe_loaded.values).mean(axis=0).max())
    # print('best id', best_k_shot_idx[0])
    # print('worst id', worst_k_shot_idx[0])
    # fig,ax = plt.subplots()
    # # ax.plot(square_score_dataframe.loc[best_k_shot_idx[0],:],label='best')
    # # ax.plot(square_score_dataframe.loc[worst_k_shot_idx[0],:],label='worst')
    # ax.scatter(square_score_dataframe.loc[best_k_shot_idx[0],:],square_score_dataframe.loc[worst_k_shot_idx[0],:])
    # ax.set_xlabel('best')
    # ax.set_ylabel('worst')
    # # ax.legend()
    # fig.savefig('plots/best_and_worst_rows.png',dpi=300)

    # fig,ax = plt.subplots()
    # # ax.plot(square_score_dataframe.loc[:,best_k_shot_idx[0]],label='best')
    # # ax.plot(square_score_dataframe.loc[:,worst_k_shot_idx[0]],label='worst')
    # to_remove = [best_k_shot_idx[0],worst_k_shot_idx[0]]
    # ax.scatter(square_score_dataframe.loc[:,best_k_shot_idx[0]].drop(to_remove),square_score_dataframe.loc[:,worst_k_shot_idx[0]].drop(to_remove))
    # ax.plot([0.8,1],[0.8,1],ls='dashed',c='black')
    # ax.set_xlabel(r'decoding $R^2$ to best $k$-shot')
    # ax.set_ylabel(r'decoding $R^2$ to worst $k$-shot')
    # # ax.set_xlabel('from_id')
    # ax.set_aspect('equal')
    # # ax.legend()
    # fig.savefig('plots/best_and_worst_columns.png',dpi=300)


    # fig,axs = plt.subplots(2,1)
    # ax = axs[0]
    # ax.scatter(square_score_dataframe.loc[:,73],square_score_dataframe.loc[:,80])
    # ax.set_xlabel(73)
    # ax.set_ylabel(80)
    # ax.set_aspect('equal')
    # # ax.set_xlabel('from_id')
    # # ax.legend()
    # ax = axs[1]
    # ax.scatter(square_score_dataframe.loc[:,53],square_score_dataframe.loc[:,18])
    # ax.set_xlabel(53)
    # ax.set_ylabel(18)
    # ax.set_aspect('equal')
    # # ax.set_xlabel('from_id')
    # # ax.legend()
    # fig.savefig('plots/selected_columns.png',dpi=300)


    # fig,axs = plt.subplots(1,2)
    # # id1 = 91
    # # id2 = 136
    # id1 = 155
    # id2 = 97
    # to_drop = [id1,id2]
    # ax = axs[0]
    # ax.scatter(square_score_dataframe.loc[:,id1].drop(to_drop),square_score_dataframe.loc[:,id2].drop(to_drop))
    # ax.set_xlabel(id1)
    # ax.set_ylabel(id2)
    # ax.set_aspect('equal')
    # ax.plot([0.78,0.9],[0.78,0.9],ls='dashed',c='black')
    # ax.set_xlabel(fr'decoding $R^2$ to {id1}')
    # ax.set_ylabel(fr'decoding $R^2$ to {id2}')
    # print(f'mean{32}shot co-bps{method_name3}')
    # print('id:',id1,select_Dscores.loc[id1,f'mean{32}shot co-bps\n{method_name3}'])
    # print('id:',id2,select_Dscores.loc[id2,f'mean{32}shot co-bps\n{method_name3}'])
    # # ax.set_xlabel('from_id')
    # # ax.legend()
    # # id1 = 68
    # # id2 = 73
    # id1 = 141
    # id2 = 69
    # to_drop = [id1,id2]
    # ax = axs[1]
    # ax.scatter(square_score_dataframe.loc[:,id1].drop(to_drop),square_score_dataframe.loc[:,id2].drop(to_drop))
    # ax.set_xlabel(fr'decoding $R^2$ to {id1}')
    # ax.set_ylabel(fr'decoding $R^2$ to {id2}')
    # ax.set_aspect('equal')
    # print(f'mean{32}shot co-bps{method_name3}')
    # print('id:',id1,select_Dscores.loc[id1,f'mean{32}shot co-bps\n{method_name3}'])
    # print('id:',id2,select_Dscores.loc[id2,f'mean{32}shot co-bps\n{method_name3}'])
    # ax.plot([0.78,0.9],[0.78,0.9],ls='dashed',c='black')
    # fig.tight_layout()
    # fig.savefig('plots/selected_columns2.png',dpi=300)


    ################### Pair grid

    # grid = sns.PairGrid(data=select_Dscores[to_plot], hue='method')
    # # grid = sns.pairplot(data = select_Dscores, hue='fewshot_method', vars=['value','co-bps', 'vel R2', 'psth R2', 'fp-bps'],diag_kind=None)
    # grid.map_upper(sns.scatterplot, s=10)
    # grid.map_lower(sns.scatterplot, s=10)
    # for ax in grid.axes[:,0]:
    #     ax.yaxis.label.set_size(5)
    # for ax in grid.axes[-1,:]:
    #     ax.xaxis.label.set_size(5)
    
    # g = grid
    # # Loop over the axes in the Pairplot
    # for i, row_axes in enumerate(g.axes):
    #     for j, ax in enumerate(row_axes):
    #         xlabel = g.x_vars[j]
    #         ylabel = g.y_vars[i]
    #         if i != j:  # Avoid diagonal plots
    #             data1 = select_Dscores[grid.x_vars[j]]
    #             data2 = select_Dscores[grid.y_vars[i]]
    #             # corr_coef = np.corrcoef(data1, data2)[0, 1]
    #             # ax.set_title(f'Correlation: {corr_coef:.2f}')
    #             ax.set_title(f'Spearman:{spearmanr(data1, data2)[0]:.2f}')
                
    #         if 'bps' in xlabel and 'bps' in ylabel and i != j:
    #             # Calculate limits based on the max of both axes for square aspect ratio
    #             xlim = ax.get_xlim()
    #             ylim = ax.get_ylim()
    #             max_limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
    #             min_limit = min(ax.get_xlim()[0], ax.get_ylim()[0])
    #             ax.plot([min_limit, max_limit], [min_limit, max_limit], ls="--", c="black")
    #             # ax.set_xlim(*[min_limit, max_limit])
    #             # ax.set_ylim(*[min_limit, max_limit])
    #             ax.set_xlim(*xlim)
    #             ax.set_ylim(*ylim)

    # # for ax in grid.axes[0,:]:
    # #     ax.set_ylim(0)
    # # for ax in grid.axes[:,0]:
    # #     ax.set_xlim(0)
    # # plt.legend(title='Method', loc='upper right', bbox_to_anchor=(1.25, 1))
    
    # handles = g.axes[0, -1].get_legend_handles_labels()[0]
    # labels = g.axes[0, -1].get_legend_handles_labels()[1]
    # # g.axes[0,-1].legend()

    # g.fig.legend(handles, labels, loc='upper right', title='Method') #
    # fig = g.fig
    # fig.suptitle('NLB metrics '+ dataset_name)
    # fig.tight_layout()
    # # fig.savefig(f'plots/NLBmetrics_{dataset_name}_best_only.png',dpi=200)
    # fig.savefig(f'plots/NLBmetrics_{dataset_name}_rates-as-latents_verybest_only_newcolsum_only.png',dpi=200)
    


#############################

for dataset_name in Dscores.dataset.unique()[:1]:
    select_Dscores = Dscores[Dscores.dataset==dataset_name]
    # print(select_Dscores[['512shot_id0 co-bps:sklearn_glm.fit_poisson_parallel','512shot_id0 co-bps:torch_glm.fit_poisson']].head(20))
    # print(select_Dscores[['sklearn_glm.fit_poisson_parallel(alpha=0.0,max_iter=500)']]) 
    # print(select_Dscores.columns)
    # fig, ax = plt.subplots()
    # sns.scatterplot(
    #     # x='512shot_id0 co-bps:sklearn_glm.fit_poisson_parallel',
    #     # y='512shot_id0 co-bps:torch_glm.fit_poisson',
    #     # x='512shot_id0 co-bps:sklearn_glm.fit_poisson_parallel(alpha=0.0,max_iter=500)',
    #     # y='512shot_id0 co-bps:sklearn_glm.fit_poisson_parallel(alpha=0.1,max_iter=500)',
    #     x='512shot_id0 co-bps:sklearn_glm.fit_poisson_parallel(alpha=0.0,max_iter=500)',
    #     y='512shot_id0 co-bps:sklearn_glm.fit_poisson_parallel(alpha=0.1,max_iter=500)',
    #     data=select_Dscores,
    #     ax=ax
    # )
    # ax.plot([0,.3],[0,.3],ls='dashed',c='black')
    # fig.savefig(f'plots/fewshotmethod_comparison_{dataset_name}.png',dpi=200)