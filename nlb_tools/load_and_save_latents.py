# import matplotlib.pyplot as plt
# from omegaconf import DictConfig, OmegaConf
# import hydra
from os import path as osp
import scipy
import json
import pandas as pd
import pickle as pkl
from multiprocessing import Pool
import numpy as np
from itertools import product
from tqdm import tqdm
from sklearn.decomposition import PCA
import scipy
from typing import Optional, Callable, Any
# from hydra.utils import instantiate
# import seaborn as sns
# import torch


from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_fewshot import extract_reallyheldout_by_id, fewshot_from_train
from nlb_tools.fewshot_utils import result_dict_to_pandas
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5, h5_to_dict
from nlb_tools.evaluation import evaluate
from nlb_tools.torch_glm import fit_poisson as fit_poisson_pl
from functools import partial
import numpy as np
import h5py
import pandas as pd

from copy import deepcopy

# import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

DATA_DIR = '/home/kabird/datasets'
 
path_to_models = '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_dmfc_rsg/240402_111015_MultiFewshot' # 200 models
latents_save_path = 'all_model_latents.h5'

threshold = 2e-3
test_train_split = 0.7

def get_full_shape(variant='mc_maze_small',target_dict=None,val_dict=None):
    if target_dict is None:
        target_path = osp.join(DATA_DIR, f"{variant}_target.h5")
        with h5py.File(target_path, 'r') as h5file:
            target_dict = h5_to_dict(h5file)

    if val_dict is None:
        val_path = osp.join(DATA_DIR, f"{variant}_val.h5")
        with h5py.File(val_path, 'r') as h5file:
            val_dict = h5_to_dict(h5file)
    
    eval_spikes_heldin = val_dict['eval_spikes_heldin']
    eval_spikes_heldout = val_dict['eval_spikes_heldout']
    eval_spikes_forward = np.concatenate(
        [target_dict[variant]['eval_spikes_heldin_forward'],target_dict[variant]['eval_spikes_heldout_forward']],
        axis = 2
    )
    eval_spikes = np.concatenate(
        [eval_spikes_heldin,eval_spikes_heldout],
        axis=-1
    )
    # print(eval_spikes.shape,eval_spikes_forward.shape,eval_spikes_heldin.shape,eval_spikes_heldout.shape)
    eval_spikes_full = np.concatenate([eval_spikes,eval_spikes_forward],axis=1)
    return eval_spikes_full

def run_model_on_numpy_dummy(
        model: Any,
        spikes_heldin: np.ndarray,
        spikes_full_shape: tuple):
    # rate_pred,factors = None, None
    print('spikes full shape',spikes_full_shape)
    trials = spikes_heldin.shape[0]
    rate_pred = np.zeros((trials, *spikes_full_shape[1:])) #spikes_full_prototype
    factors = np.zeros((trials, *spikes_full_shape[1:]))
    return rate_pred,factors

def load_nlb_data(variant='mc_maze_small'):
    target_path = osp.join(DATA_DIR, f"{variant}_target.h5")

    with h5py.File(target_path, 'r') as h5file:
        target_dict = h5_to_dict(h5file)

    
    train_path = osp.join(DATA_DIR, f"{variant}_train.h5")
    with h5py.File(train_path, 'r') as h5file:
        train_dict = h5_to_dict(h5file)
    
    val_path = osp.join(DATA_DIR, f"{variant}_val.h5")
    with h5py.File(val_path, 'r') as h5file:
        val_dict = h5_to_dict(h5file)

    train_path_json = osp.join(DATA_DIR, f"{variant}_train.json")
    with open(train_path_json, 'r') as file:
        few_shot_metadata = json.load(file)
    
    return  train_dict,val_dict,target_dict,few_shot_metadata

def run_nlb_evaluation_protocol(
    model: Any,
    run_model_on_numpy_pre: Callable,
    variant='mc_maze_small',
    do_fewshot: bool = False,
    do_evaluation: bool = True,
    ):
    """
    Evaluation protocol:
    Load data dicts.
    Run model on validation set.
    Run model on fewshot trials, get latents, train fewshot head.
    """

    train_dict,val_dict,target_dict,few_shot_metadata = load_nlb_data(variant=variant)

    
    # print(few_shot_metadata)
    

    # model()
    batch_size = 8

    eval_spikes_full_shape = get_full_shape(variant=variant,target_dict=target_dict,val_dict=val_dict).shape


    eval_spikes_heldin = val_dict['eval_spikes_heldin']
    eval_spikes_heldout = val_dict['eval_spikes_heldout']
    train_spikes_heldin = train_dict['train_spikes_heldin']
    run_model_on_numpy = partial(run_model_on_numpy_pre,spikes_full_shape=eval_spikes_full_shape)
    
    eval_pred, eval_latents = run_model_on_numpy(model,eval_spikes_heldin)
    train_pred, _ = run_model_on_numpy(model,train_spikes_heldin)

    print('eval pred shape',eval_pred.shape)
    eval_latents = eval_latents[:,:eval_spikes_heldin.shape[1]]
    eval_latents_s = eval_latents.reshape(-1,eval_latents.shape[-1])
    

    train_rates,eval_rates = train_pred,eval_pred
    spikes = eval_spikes_heldin
    heldout_spikes = eval_spikes_heldout

    eval_rates, eval_rates_forward = np.split(eval_rates, [spikes.shape[1]], axis=1)
    eval_rates_heldin_forward, eval_rates_heldout_forward = np.split(eval_rates_forward, [spikes.shape[-1]], axis=-1)
    train_rates, _ = np.split(train_rates, [spikes.shape[1]], axis=1)
    eval_rates_heldin, eval_rates_heldout = np.split(eval_rates, [spikes.shape[-1]], axis=-1)
    train_rates_heldin, train_rates_heldout = np.split(train_rates, [spikes.shape[-1]], axis=-1)

    latents_dict = {
        variant: {
            'eval_latents': eval_latents,
        }
    }

    output_dict = {
        variant: {
            'train_rates_heldin': train_rates_heldin,
            'train_rates_heldout': train_rates_heldout,
            'eval_rates_heldin': eval_rates_heldin,
            'eval_rates_heldout': eval_rates_heldout,
            'eval_rates_heldin_forward': eval_rates_heldin_forward,
            'eval_rates_heldout_forward': eval_rates_heldout_forward
        }
    }
    for key,val in output_dict[variant].items():
        print(key,val.shape)


    fewshot_output_dict = {}
    fewshot_latents_dict = {}
    if do_fewshot:
        k_range = few_shot_metadata["Kvalues_applicable"][-2:-1] #2**np.arange(4,11)[:1].astype(int)
        # k_range = [int(k) for k in k_range]
        for k in k_range[:]:
            for shot_id in few_shot_metadata[f'{k}shot_ids'][:]:
                fewshot_train_spikes_heldin = train_dict[f'{k}shot_id{shot_id}_train_spikes_heldin']
                fewshot_train_spikes_heldout = train_dict[f'{k}shot_id{shot_id}_train_spikes_heldout']
                fewshot_train_spikes_heldout_s = fewshot_train_spikes_heldout.reshape(-1,fewshot_train_spikes_heldout.shape[-1])
                
                fewshot_train_pred, fewshot_train_latents = run_model_on_numpy(model,fewshot_train_spikes_heldin)
                fewshot_train_latents = fewshot_train_latents[:,:fewshot_train_spikes_heldin.shape[1]]
                # fewshot_train_latents_s = fewshot_train_latents.reshape(-1,fewshot_train_latents.shape[-1])
                # fewshot_train_rates_s, eval_rates_s = fit_poisson_pl(fewshot_train_latents_s,eval_latents_s,fewshot_train_spikes_heldout_s)
                # eval_rates = eval_rates_s.reshape(*heldout_spikes.shape[:2],-1)
                # fewshot_output_dict [f'{k}shot_id{shot_id}_eval_rates_heldout'] = eval_rates
                fewshot_latents_dict[f'{k}shot_id{shot_id}_train_latents'] = fewshot_train_latents

    output_dict[variant] = {
        **output_dict[variant],
        **fewshot_output_dict
    }
    
    latents_dict[variant] = {
        **latents_dict[variant],
        **fewshot_latents_dict
    }

    fewshot_code_name = 'sklearn_parallel'

    result_data_dict = None
    result_data_df = None
    if do_evaluation:
        result_data_dict = evaluate(target_dict, output_dict)
        print('result_dict',result_data_dict)
        result_data_df = result_dict_to_pandas(
            result_data_dict,
            fewshot_learner=fewshot_code_name,
            # path=ckpt_path
        )

    # eval_report.append(df)
    
    # D = result_data.reset_index()
    # D.to_csv()
    return result_data_df, result_data_dict, latents_dict, output_dict

def run_fewshot_given_latents(
        latents_dict,
        result_save_path='',
        variant='mc_maze_small',
        do_evaluation=True,
        output_dict: dict = {},
        fit_poisson_func: Callable = fit_poisson_pl):
    train_dict,val_dict,target_dict,few_shot_metadata = load_nlb_data(variant=variant)
                                                         
    eval_latents = latents_dict[variant]['eval_latents']
    eval_latents_s = eval_latents.reshape(-1,eval_latents.shape[-1])
    heldout_spikes = val_dict['eval_spikes_heldout']

    fewshot_output_dict = {}

    k_range = few_shot_metadata["Kvalues_applicable"] #2**np.arange(4,11)[:1].astype(int)
    # k_range = [int(k) for k in k_range]
    for k in k_range[:]:
        for shot_id in few_shot_metadata[f'{k}shot_ids']:
            # fewshot_train_spikes_heldin = train_dict[f'{k}shot_id{shot_id}_train_spikes_heldin']
            fewshot_train_spikes_heldout = train_dict[f'{k}shot_id{shot_id}_train_spikes_heldout']
            fewshot_train_spikes_heldout_s = fewshot_train_spikes_heldout.reshape(-1,fewshot_train_spikes_heldout.shape[-1])
            
            if f'{k}shot_id{shot_id}_train_latents' in latents_dict[variant]:
                fewshot_train_latents = latents_dict[variant][f'{k}shot_id{shot_id}_train_latents']
                fewshot_train_latents_s = fewshot_train_latents.reshape(-1,fewshot_train_latents.shape[-1])
                
                fewshot_train_rates_s, eval_rates_s = fit_poisson_func(fewshot_train_latents_s,eval_latents_s,fewshot_train_spikes_heldout_s)
                eval_rates = eval_rates_s.reshape(*heldout_spikes.shape[:2],-1)
                fewshot_output_dict [f'{k}shot_id{shot_id}_eval_rates_heldout'] = eval_rates

    if variant not in output_dict.keys():
        output_dict[variant] = {}
    output_dict[variant].update({**fewshot_output_dict})
    
    # fewshot_code_name = 'sklearn'
    result_data_dict = None
    result_data_df = None
    if do_evaluation:
        # print(output_dict)
        result_data_dict = evaluate(target_dict, output_dict)
        print('result_dict',result_data_dict)
        result_data_df = result_dict_to_pandas(
            result_data_dict,
            # fewshot_learner=fewshot_code_name,
            path=result_save_path
        )

    # eval_report.append(df)
    
    # D = result_data.reset_index()
    # D.to_csv()
    return result_data_df, result_data_dict

    

if __name__=="__main__":
    run_nlb_evaluation_protocol(
        "",run_model_on_numpy_pre=run_model_on_numpy_dummy,variant='mc_maze'
    )
