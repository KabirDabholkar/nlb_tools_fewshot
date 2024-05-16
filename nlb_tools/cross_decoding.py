import matplotlib.pyplot as plt
# from omegaconf import DictConfig, OmegaConf
# import hydra
import os
import scipy
import pandas as pd
import pickle as pkl
from multiprocessing import Pool
import numpy as np
from itertools import product
from tqdm import tqdm
from sklearn.decomposition import PCA
import scipy
from typing import Optional
# from hydra.utils import instantiate
# from hydra.utils import instantiate
import seaborn as sns
import torch
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path

from typing import Callable, Optional

from nlb_tools.make_tensors import h5_to_dict
import h5py
from functools import partial

from copy import deepcopy

import matplotlib as mpl


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
    #     'path' : '/home/kabird/STNDT_fewshot/ray_results/mc_maze_20_lite_all6.csv', # STNDT mc_maze
    #     'variant' : 'mc_maze_20',
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
def load_result_files(result_files):
    D = []
    for f in result_files:
        d_ = pd.read_csv(f['path'],index_col=0)
        d_['method'] = f['method']
        # print(d_.head(2).iloc[:,:5])   
        D.append(d_)

    D = pd.concat(D,axis=0)
    return D


def select_best_cobps(D,use='co-bps',margin = 2e-2):
    threshold_cobps = D[use].max() - margin
    Dbest = D[D[use]>threshold_cobps]
    return Dbest

# def train_test_split(D:pd.DataFrame):
#     for i,row in D.iterrows():
#         row['latents']



def load_latents(D,process = None,test_only_one=False):
    latents = []
    for i,row in D.iterrows():
        result_path = row['path']
        latent_path = result_path.split('results_all')[0] + 'latents.h5'
        latent_path = result_path.split('results_new')[0] + 'latents.h5'
        # path_exists = os.path.exists(latent_path)
        try:
            with h5py.File(latent_path, 'r') as h5file:
                latent = h5_to_dict(h5file)
        except OSError:
            latent = None
        if latent is not None:
            if process is not None:
                latent = process(latent)
        latents.append( latent )
        if test_only_one:
            return latents
        
    return latents

def extract_latent(data_dict):
        latent = list(data_dict.values())[0]
        latent = latent['eval_latents']
        return latent

def train_model(model, dataset):
    """Function to train a regression model"""
    X, y = dataset
    print(X.shape,y.shape)
    model.fit(X, y)
    return model



def score_model(model, dataset, metric, predict_method):
    X, y = dataset
    pred_y = getattr(model, predict_method)(X)
    # score = np.stack([metric(
    #     y[sample_id],
    #     pred_y[sample_id]
    # ) for sample_id in range(pred_y.shape[0])]).mean()
    print(y.shape,pred_y.shape)
    score = metric(y,pred_y)
    return score

def cross_decoding(
        best_latents_dataframe,
        preprocess_target: Optional[Callable] = None,
        regression_model: BaseEstimator = None,
        metric: Callable = None,
        predict_method: str = 'predict',
        # primary_metric: str = 'valid/co_bps',
    ):
    # omegaconf_resolvers()

    # saveloc = os.path.join(path_to_models, 'concat_model_data.csv')
    # latents_dataframe = pd.read_csv(saveloc)
    # latents_dataframe = load_latents(latents_dataframe)

    # best_latents_dataframe = latents_dataframe[latents_dataframe[primary_metric] > (latents_dataframe['valid/co_bps'].max() - 2e-2)]
    
    # best_latents_dataframe = best_latents_dataframe.head(2)

    n_models = len(best_latents_dataframe)

    print('n_models:', n_models)

    train_latents = best_latents_dataframe['train_latents'].values
    train_latents_r = [thing.reshape(-1, thing.shape[-1]) for thing in train_latents]
    if preprocess_target is not None:
        train_latents_r = [preprocess_target(thing) for thing in train_latents_r]
    train_datasets = []
    for i, j in tqdm(list(product(range(n_models), range(n_models)))):
        train_datasets.append(
            (train_latents_r[i], train_latents_r[j])
        )
    test_latents = best_latents_dataframe['test_latents'].values
    print('len(test_latents)',len(test_latents))
    test_latents_r = [thing.reshape(-1, thing.shape[-1]) for thing in test_latents]
    test_datasets = []
    for i, j in tqdm(list(product(range(n_models), range(n_models)))):
        test_datasets.append(
            (test_latents_r[i], test_latents_r[j])
        )
    models = [
        regression_model()
        for m in range(len(train_datasets))
    ]
    # metric = 
    with Pool() as p:
        trained_models = p.starmap(train_model, zip(models, train_datasets))
        scores = p.starmap(
            score_model,
            zip(
                trained_models,
                # test_datasets,
                test_datasets,
                # [metric] * len(train_datasets),
                # [predict_method] * len(train_datasets)
                [metric] * len(test_datasets),
                [predict_method] * len(test_datasets)
            )
        )

    print(len(scores))
    score_dataframe = pd.DataFrame({
        'from_to_index' : list(product(range(n_models), range(n_models))),
        'score' : scores,
    })
    score_dataframe[['from','to']]=pd.DataFrame(score_dataframe['from_to_index'].to_list(),index=score_dataframe.index)
    print(score_dataframe)
    score_dataframe['from_id'] = best_latents_dataframe['model_id'].iloc[score_dataframe['from']].values
    score_dataframe['to_id'] = best_latents_dataframe['model_id'].iloc[score_dataframe['to']].values
    # saveloc = os.path.join(saveloc, 'cross_decoding_scores.csv')
    # score_dataframe.to_csv(saveloc)
    
    scores = np.array(scores).reshape(n_models, n_models)
    # saveloc = os.path.join(savedir, 'cross_decoding_scores_parallel')
    # np.save(saveloc, scores)
    return score_dataframe, scores, trained_models, test_datasets

def compute_colsums(best_results,score_dataframe):
    
    # additionally condition on good heldin co_bps
    # latents_dataframe = latents_dataframe[latents_dataframe[f'debugging/val_{1000}shot_co_bps_recon_truereadout']>0.36]
    # latents_dataframe = latents_dataframe[latents_dataframe['valid/co_bps'] > (latents_dataframe['valid/co_bps'].max() - 1e-2)]
    # latents_dataframe = latents_dataframe[latents_dataframe.co_bps > (latents_dataframe.co_bps.max() - 1e-2)]
    
    # saveloc = os.path.join(path_to_models, 'cross_decoding_scores.csv')
    # score_dataframe = pd.read_csv(saveloc)
    latents_dataframe = best_results
    #score_dataframe['from_id'] = score_dataframe['from_id'].str.split('_').str[-1]
    #score_dataframe['to_id']   = score_dataframe['to_id'].str.split('_').str[-1]
    print(score_dataframe['from_id'])
    #.drop(['from_to_index','from','to'],axis=1)
    square_score_dataframe = score_dataframe.pivot(
        index = 'from_id',
        columns = 'to_id',
        values = 'score'
    )
    print(len(latents_dataframe.index.values))
    select_ids = latents_dataframe.index
    select_ids = select_ids.intersection(square_score_dataframe.index)
    selection = square_score_dataframe.loc[select_ids].index
    best_results = best_results.loc[select_ids]
    # print(square_score_dataframe)
    square_score_dataframe = square_score_dataframe.loc[selection,:].loc[:,selection]
    square_dist_dataframe = 1 - np.array(square_score_dataframe.values)
    col_sums = square_dist_dataframe.mean(axis=0)
    best_results['1-R^2 column sum'] = col_sums
    return best_results


if __name__=="__main__":

    D = load_result_files(result_files)
    print(D['co-bps'])

    print(D.shape)
    D = select_best_cobps(D,use='valid/co_bps',margin=1e-2) #2e-2
    
    latents = load_latents(D,test_only_one=False,process=extract_latent)
    # latent = latents[0]
    # print(latent.shape)
    D['latents'] = latents
    D.dropna(subset=['latents'],inplace=True)
    
    # D = D.head(3)
    print(D.shape)
    
    # # from sklearn.decomposition import PCA
    # for id,lat in enumerate(D['latents']):
    #     factors = lat

    #     lat_from = D['latents'].iloc[0]
    #     reg_model = LinearRegression()
    #     reg_model.fit(
    #         lat_from.reshape(-1,lat_from.shape[-1]),
    #         lat.reshape(-1,lat.shape[-1]),
    #     )
    #     lat_pred = reg_model.predict(lat_from.reshape(-1,lat_from.shape[-1])).reshape(lat.shape)
    #     score_from_func = score_model(reg_model,
    #                  (lat_from.reshape(-1,lat_from.shape[-1]),lat.reshape(-1,lat.shape[-1])),
    #                  partial(r2_score,multioutput='uniform_average'), 
    #                  'predict')
    #     print(f'r2 {id}',r2_score(lat.reshape(-1,lat.shape[-1]),lat_pred.reshape(-1,lat_pred.shape[-1])))
    #     print(f'r2 func {id}',score_from_func)

    #     print('factors shape',factors.shape)
    #     n_components = 2
    #     P = PCA(n_components=n_components)
    #     factors_proj =P.fit_transform(factors.reshape(-1,factors.shape[-1])).reshape(*factors.shape[0:2],n_components)
    #     fig,ax = plt.subplots()
    #     for i in range(10):
    #         ax.plot(factors_proj[i,:,0],factors_proj[i,:,1],c=f'C{i}')

    #     pred_proj = P.transform(lat_pred.reshape(-1,lat_pred.shape[-1])).reshape(*lat_pred.shape[:2],n_components)
        
    #     for i in range(10):
    #         ax.plot(pred_proj[i,:,0],pred_proj[i,:,1],ls='dashed',c=f'C{i}')
    #     fig.savefig(f'/home/kabird/nlb_tools_fewshot/plots/PCA_nlb{id}.png',dpi=300)

    

    D['train_latents'],D['test_latents'] = zip(*D['latents'].apply(partial(train_test_split,random_state=0)))
    D['model_id'] = D['path']
    

    scores_dataframe,scores,trained_models,test_datasets = cross_decoding(
        best_latents_dataframe=D,
        preprocess_target=None,
        regression_model=LinearRegression,
        metric=partial(r2_score,multioutput='uniform_average'),
    )
    # lat0 = D['train_latents'].iloc[0]
    # lat1 = D['train_latents'].iloc[1]
    # reg_model = LinearRegression()
    # reg_model.fit(
    #     lat0.reshape(-1,lat0.shape[-1]),lat1.reshape(-1,lat1.shape[-1])
    # )
    # lat0 = D['latents'].iloc[0]
    # lat1 = D['latents'].iloc[1]
    # reg_model2 = LinearRegression()
    # reg_model2.fit(
    #     lat0.reshape(-1,lat0.shape[-1]),lat1.reshape(-1,lat1.shape[-1])
    # )
    # lat0 = D['latents'].iloc[0]
    # lat1 = D['latents'].iloc[1]
    # print('trained model score',score_model(reg_model2,
    #                                         (lat0.reshape(-1,lat0.shape[-1]),lat1.reshape(-1,lat1.shape[-1])),
    #                                         r2_score,'predict'))
    # lat0 = D['test_latents'].iloc[0]
    # lat1 = D['test_latents'].iloc[1]
    # print('trained model score',score_model(reg_model,
    #                                         (lat0.reshape(-1,lat0.shape[-1]),lat1.reshape(-1,lat1.shape[-1])),
    #                                         r2_score,'predict'))
    # print('trained model score',score_model(trained_models[1],
    #                                         (lat0.reshape(-1,lat0.shape[-1]),lat1.reshape(-1,lat1.shape[-1])),
    #                                         r2_score,'predict'))
    # print('trained model score',score_model(trained_models[1],test_datasets[1],r2_score,'predict'))

    result_path = result_files[0]['path']
    pathdir = result_path.split('.csv')[0]
    csv_path = pathdir + '_cross_decoding_scores.csv'
    scores_dataframe.to_csv(csv_path)

    numpy_path = pathdir + '_cross_decoding_scores'
    np.save(numpy_path, scores)


    
    
    
        


    
    

