import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.decomposition import PCA
import os
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as patches

from nlb_tools.cross_decoding import load_result_files,load_latents,select_best_cobps,extract_latent,compute_colsums

mpl.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

result_files = [
    # {
    #     'method'  : 'STNDT',
    #     'path' : '/home/kabird/STNDT_fewshot/results5.csv', # STNDT mc_maze
    # },
    # {
    #     'method'  : 'STNDT',
    #     'path' : '/home/kabird/STNDT_fewshot/ray_results/mc_maze_20_lite_all5.csv', # STNDT mc_maze
    # },
    # {
    #     'method' : 'lfads-torch',
    #     'path' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_dmfc_rsg/240501_030533_MultiFewshot/concatenated_results.csv', # LFADS dmfc_rsg
    # }
    {
        'method' : 'lfads-torch',
        'path' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240514_000059_MultiFewshot/concatenated_results.csv', # LFADS mc_maze_20 22 really heldout
    },
    
]




def plot_cross_decoding_scores(path_to_scores_numpy,path_to_savefig):
    saveloc = path_to_scores_numpy
    scores = np.load(saveloc)
    print(scores.shape)

    fig, ax = plt.subplots()
    im = ax.imshow(1-scores)
    ax.set_ylabel('input model')
    ax.set_xlabel('target model')
    fig.colorbar(im, ax=ax, label=r'$1-R^2$')
    fig.tight_layout()
    fig.savefig(path_to_savefig, dpi=300)

    plt.close(fig)


def plot_cross_decoding_scores_from_dataframe(
        score_dataframe,
        select_indices = None,
        highlight_rows=[],
        highlight_cols=[],
        figsavepath = 'plots/cross_decoding_with_highlights.png',
        style = 'arrow'
        ):
    square_score_dataframe = score_dataframe.pivot(
        index = 'from_id',
        columns = 'to_id',
        values = 'score'
    )
    df = square_score_dataframe.apply(lambda x: 1-x)
    fig,ax= plt.subplots(figsize=(10,10))
    if select_indices is not None:
        df = df.loc[select_indices,select_indices]
    sns.heatmap(df, annot=False, cmap="viridis", cbar=True, linewidths=.5, ax=ax)
    ax.set_aspect('equal')
    # ax.imshow(df.values)
    if style == 'rect':
        for idx in highlight_rows:
            plt.gca().add_patch(plt.Rectangle((0, idx), len(df.columns), 1, fill=False, edgecolor='black', lw=1, alpha=0.5))
        for idx in highlight_cols:
            plt.gca().add_patch(plt.Rectangle((idx, 0), 1, len(df), fill=False, edgecolor='black', lw=1, alpha=0.5))
    elif style == 'arrow':
        ax = plt.gca()

        # Adding arrows for highlighted rows
        for idx in highlight_rows:
            ax.add_patch(FancyArrowPatch((len(df.columns), idx + 0.5), (len(df.columns) + 0.3, idx + 0.5),
                                        color="red", arrowstyle="->,head_width=0.5,head_length=0.3"))

        # Adding arrows for highlighted columns
        for idx in highlight_cols:
            ax.add_patch(FancyArrowPatch((idx + 0.5, len(df)), (idx + 0.5, len(df) + 0.3),
                                        color="blue", arrowstyle="->,head_width=0.5,head_length=0.3"))
    elif style== 'triangle':
        # # Adding triangle markers on the outer spines
        triangle_size = 0.5  # Size of the triangle marker
        for idx in highlight_rows:
            ax.add_patch(patches.RegularPolygon((0 - triangle_size, idx + 0.5), numVertices=3, radius=triangle_size, orientation=np.pi / 2, color='black'))
            ax.add_patch(patches.RegularPolygon((len(df.columns) + triangle_size - 1, idx + 0.5), numVertices=3, radius=triangle_size, orientation=3 * np.pi / 2, color='black'))

        for idx in highlight_cols:
            ax.add_patch(patches.RegularPolygon((idx + 0.5, 0 - triangle_size), numVertices=3, radius=triangle_size, orientation=0, color='black'))
            ax.add_patch(patches.RegularPolygon((idx + 0.5, len(df) + triangle_size - 1), numVertices=3, radius=triangle_size, orientation=np.pi, color='black'))
        # Adding triangle markers on the right and top spines
        # triangle_size = 0.3  # Size of the triangle marker

        # # Right spine - Rows
        # for idx in highlight_rows:
        #     ax.add_patch(patches.RegularPolygon((len(df.columns) + triangle_size, idx + 0.5), numVertices=3, radius=triangle_size, orientation=0, color='red'))

        # # Top spine - Columns
        # for idx in highlight_cols:
        #     ax.add_patch(patches.RegularPolygon((idx + 0.5, -triangle_size), numVertices=3, radius=triangle_size, orientation=np.pi/2, color='blue'))
    else:
        raise Exception("'style' must be one of 'rect' or 'arrow'")
    fig.savefig(figsavepath,dpi=300)


def check_extremes(best_results_with_latents):
    colmean_sorted = best_results_with_latents['1-R^2 column sum'].sort_values(ascending=True)
    latents_dataframe = best_results_with_latents

    min_colmean = colmean_sorted.head(1) # what we think should be the good model
    max_colmean = colmean_sorted.tail(1) # what we think should be the bad model
    # print(
    #     min_colmean.index,
    #     max_colmean.index,
    #     colmean_sorted
    # )
    print('colmean_sorted.index',colmean_sorted.index)
    # print(
    #     'From min to max col-mean model 1-R2',1-square_score_dataframe.loc[min_colmean.index,max_colmean.index].values,'\n'
    #     'From max to min col-mean model 1-R2',1-square_score_dataframe.loc[max_colmean.index,min_colmean.index].values
    # )

    n_components = 3
    num_trials_to_plot = 5
    for name,model_id in zip(['max_colmean'],[pd.Index([161])] ):#['min_colmean','max_colmean'],[min_colmean.index,max_colmean.index]
        id = model_id  #.values[0].split('_')[-1]
        print(name,'id:',model_id,colmean_sorted.loc[model_id])
        # print([thing.shape for thing in latents_dataframe.loc[id][['train_latents','test_latents']].values])
        latents = latents_dataframe.loc[id]['latents'].values[0]
        P = PCA(n_components=n_components)
        latents_pcproj = P.fit_transform(latents.reshape(-1,latents.shape[-1])).reshape(*latents.shape[:2],n_components)

        fig = plt.figure()
        ax = fig.add_subplot()#projection='3d')
        print([t.shape for t in latents_pcproj[...,:3].swapaxes(0,-1).swapaxes(1,2)])
        for i in range(num_trials_to_plot):
            ax.plot(latents_pcproj[i,...,0].T,latents_pcproj[i,...,1].T,lw=1.4,alpha=0.7)#,latents_pcproj[...,2])
            ax.scatter(latents_pcproj[i,0,0],latents_pcproj[i,0,1],c='green',s=10)
            ax.scatter(latents_pcproj[i,-1,0],latents_pcproj[i,-1,1],c='red',s=10)
        ax.axis('off')
        saveloc = os.path.join('plots', name+'_PCA.png')
        fig.tight_layout()
        fig.savefig(saveloc,dpi=300)
        plt.close()
        # return fig


    # data = []
    # for model_id in colmean_sorted.index[:]:
    #     id = model_id.split('_')[-1]
    #     latents = np.concatenate(latents_dataframe.loc[id][['train_latents','test_latents']].values,axis=0)
    #     p = PCA()
    #     p.fit(latents.reshape(-1,latents.shape[-1]))
    #     data.append(pd.DataFrame({
    #         'variance explained':p.explained_variance_ratio_,
    #         'model_id':model_id,
    #         'components':np.arange(p.n_components_),
    #         'decoding_colmean':[colmean_sorted.loc[model_id]]*p.n_components_,
    #     }))

    # data = pd.concat(data,axis=0)
    
    # fig,ax = plt.subplots()
    # sns.lineplot(x='components',y='variance explained',units='model_id',hue='decoding_colmean',data=data,ax=ax)
    # ax.set_yscale('log')
    # # ax.set_xlim(0,20)
    # # ax.set_ylim(1e-2,2e-1)
    # fig.tight_layout()
    # saveloc = os.path.join(path_to_models, 'variance_explained.png')
    # fig.savefig(saveloc,dpi=300)


if __name__ == "__main__":
    # for [0]['path'].replace('.csv','_with_means.csv')
    # D = load_result_files(result_files)
    D = pd.read_csv(result_files[0]['path'].replace('.csv','_with_means.csv'))

    # print(D['co-bps'])

    # print(D.shape)
    D = select_best_cobps(D,margin=0.7e-2)
    
    # latents = load_latents(D,test_only_one=False,process=extract_latent)
    # # latent = latents[0]
    # # print(latent.shape)
    # D['latents'] = latents
    # D.dropna(subset=['latents'],inplace=True)


    
    # results = pd.read_csv(
    #     '/home/kabird/STNDT_fewshot/ray_results/mc_maze_20_lite_all4.csv'
    # )
    


    plot_cross_decoding_scores(
        path_to_scores_numpy = result_files[0]['path'].replace('.csv','_cross_decoding_scores.npy'),
        path_to_savefig = f'plots/cross_decoding_{result_files[0]["method"]}_mc_maze_20.png' 
    )

    scores = pd.read_csv(result_files[0]['path'].replace('.csv','_cross_decoding_scores.csv'))
    scores_old = pd.read_csv(os.path.join(os.path.dirname(result_files[0]['path']), 'cross_decoding_scores_old.csv'),index_col=0)
    print(scores_old.columns)
    scores_old['from_id'] = scores_old['from_id'].str.split('_').str[3].astype(int)
    scores_old['to_id'] = scores_old['to_id'].str.split('_').str[3].astype(int)
    scores_old.drop(columns=['from_to_index','from','to'],inplace=True)
    

    # print('shapes',D.shape,len(D['path'].unique()))
    D.index = D['path'].str.split('/').str[7].str.split('_').str[3].astype(int)
    scores['from_id'] = scores['from_id'].str.split('/').str[7].str.split('_').str[3].astype(int)
    scores['to_id'] = scores['to_id'].str.split('/').str[7].str.split('_').str[3].astype(int)
    scores.drop(columns=['from_to_index','from','to'],inplace=True)
    
    scores_old_and_new = pd.merge(scores, scores_old, on=['from_id', 'to_id'],suffixes=('_new','_old'))
    

    print(scores.shape)
    print(scores_old.shape)

    print('scores_old_and_new',scores_old_and_new.columns)

    fig,ax = plt.subplots()
    # ax.scatter(scores_old_and_new['score_old'],scores_old_and_new['score_new'])
    sns.scatterplot(x='score_old',y='score_new',data=scores_old_and_new,ax=ax,s=10)
    ax.plot([0.75,1],[0.75,1],ls='dashed',c='black')
    
    ax.set_title(f'corrcoef{np.corrcoef(scores_old_and_new["score_old"],scores_old_and_new["score_new"])[0,1]}')
    fig.savefig('plots/cross_decoding_scores_old_and_new.png')

    # print(D.index)
    # print(scores['from_id'])

    best_results = D
    best_results = compute_colsums(
        best_results,
        scores
    )

    # check_extremes(best_results)
    # print(best_results)
    best_results.dropna(subset=['mean64shot co-bps:torch_glm.fit_poisson'],inplace=True)

    # fig,axs = plt.subplots(1,4,figsize=(15,4))
    # ax = axs[0]
    # sns.scatterplot(
    #     data = best_results,
    #     x='1-R^2 column sum',
    #     # y='mean512shot co-bps:sklearn_glm.fit_poisson_parallel(alpha=0.1,max_iter=500)',
    #     # y='mean64shot co-bps:sklearn_glm.fit_poisson_parallel(alpha=0.1,max_iter=500)',
    #     y='mean64shot co-bps:torch_glm.fit_poisson',
    #     ax = ax,
    # )
    # ax.set_title(np.corrcoef(
    #     best_results['1-R^2 column sum'],
    #     best_results['mean64shot co-bps:torch_glm.fit_poisson']
    #     )[0,-1]
    # )
    # ax = axs[1]
    # sns.scatterplot(
    #     data = best_results,
    #     x='1-R^2 column sum',
    #     # y='mean512shot co-bps:sklearn_glm.fit_poisson_parallel(alpha=0.1,max_iter=500)',
    #     y='mean64shot co-bps:sklearn_glm.fit_poisson_parallel(alpha=0.01,max_iter=500)',
    #     ax = ax,
    # )
    # ax.set_title(np.corrcoef(
    #     best_results['1-R^2 column sum'],
    #     best_results['mean64shot co-bps:sklearn_glm.fit_poisson_parallel(alpha=0.01,max_iter=500)']
    #     )[0,-1]
    # )
    # ax = axs[2]
    # sns.scatterplot(
    #     data = best_results,
    #     x='1-R^2 column sum',
    #     # y='mean512shot co-bps:sklearn_glm.fit_poisson_parallel(alpha=0.1,max_iter=500)',
    #     y='mean64shot co-bps:sklearn_glm.fit_poisson_parallel(alpha=0.001,max_iter=500)',
    #     ax = ax,
    # )
    # ax.set_title(np.corrcoef(
    #     best_results['1-R^2 column sum'],
    #     best_results['mean64shot co-bps:sklearn_glm.fit_poisson_parallel(alpha=0.001,max_iter=500)']
    #     )[0,-1]
    # )
    # ax = axs[3]
    # sns.scatterplot(
    #     data = best_results,
    #     x='1-R^2 column sum',
    #     y='co-bps',
    #     ax = ax,
    # )
    # ax.set_title(np.corrcoef(
    #     best_results['1-R^2 column sum'],
    #     best_results['co-bps']
    #     )[0,-1]
    # )
    # for ax in axs:
    #     ax.set_xlabel(r'$1-R^2$ column sum')
    # fig.tight_layout()
    # fig.savefig('plots/colsums_vs_cobps_and_fewshot.png',dpi=300)
    # print('done')