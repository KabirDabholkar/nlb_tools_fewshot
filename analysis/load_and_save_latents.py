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
import seaborn as sns
import torch

from copy import deepcopy

# import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

path_to_models = '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_dmfc_rsg/240402_111015_MultiFewshot' # 200 models
latents_save_path = 'all_model_latents.h5'

threshold = 2e-3
test_train_split = 0.7
