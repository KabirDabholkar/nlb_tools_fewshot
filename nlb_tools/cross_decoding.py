import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import hydra
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

