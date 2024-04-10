# ---- Imports ---- #
import numpy as np
import pandas as pd
import h5py
import neo
import quantities as pq
from elephant.gpfa import GPFA
from sklearn.linear_model import LinearRegression, PoissonRegressor, Ridge
from functools import partial

from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
from nlb_tools.make_fewshot import extract_reallyheldout_by_id, fewshot_from_train
from nlb_tools.evaluation import evaluate

# ---- Default params ---- #
default_dict = { # [latent_dim, alpha1, alpha2]
    'mc_maze': [52, 0.01, 0.0],
    'mc_rtt': [36, 0.0, 0.0],
    'area2_bump': [22, 0.0001, 0.0],
    'dmfc_rsg': [32, 0.0001, 0.0],
    'mc_maze_large': [44, 0.01, 0.0],
    'mc_maze_medium': [28, 0.0, 0.0],
    'mc_maze_small': [18, 0.01, 0.0],
}

# ---- Run Params ---- #
dataset_name = "mc_maze" # one of {'area2_bump', 'dmfc_rsg', 'mc_maze', 'mc_rtt', 
                            # 'mc_maze_large', 'mc_maze_medium', 'mc_maze_small'}
bin_size_ms = 20
Kvalues = (2**np.arange(2,15)).astype(int)
really_heldout_neurons_ids = np.arange(10,dtype=int)
extract_reallyheldout_by_id_partial = partial(extract_reallyheldout_by_id,neuron_ids_to_extract=really_heldout_neurons_ids)
# replace defaults with other values if desired
latent_dim = default_dict[dataset_name][0]
alpha1 = default_dict[dataset_name][1]
alpha2 = default_dict[dataset_name][2]
phase = 'val' # one of {'test', 'val'}

# ---- Useful variables ---- #
binsuf = '' if bin_size_ms == 5 else f'_{bin_size_ms}'
dskey = f'mc_maze_scaling{binsuf}_split' if 'maze_' in dataset_name else (dataset_name + binsuf + "_split")
pref_dict = {'mc_maze_small': '[100] ', 'mc_maze_medium': '[250] ', 'mc_maze_large': '[500] '}
bpskey = pref_dict.get(dataset_name, '') + 'co-bps'

# ---- Data locations ---- #
datapath_dict = {
    'mc_maze': '~/datasets/000128/sub-Jenkins/',
    'mc_rtt': '~/datasets/000129/sub-Indy/',
    'area2_bump': '~/datasets/000127/sub-Han/',
    'dmfc_rsg': '~/datasets/000130/sub-Haydn/',
    'mc_maze_large': '~/datasets/000138/sub-Jenkins/',
    'mc_maze_medium': '~/datasets/000139/sub-Jenkins/',
    'mc_maze_small': '~/datasets/000140/sub-Jenkins/',
}
prefix_dict = {
    'mc_maze': '*full',
    'mc_maze_large': '*large',
    'mc_maze_medium': '*medium',
    'mc_maze_small': '*small',
}
datapath = datapath_dict[dataset_name]
prefix = prefix_dict.get(dataset_name, '')
savepath = f'{dataset_name}{"" if bin_size_ms == 5 else f"_{bin_size_ms}"}_gpfa_output_{phase}.h5'

# ---- Load data ---- #
dataset = NWBDataset(datapath, prefix, 
    skip_fields=['hand_pos', 'cursor_pos', 'eye_pos', 'muscle_vel', 'muscle_len', 'joint_vel', 'joint_ang', 'force'])
dataset.resample(bin_size_ms)

# ---- Extract data ---- #
if phase == 'val':
    train_split = 'train'
    eval_split = 'val'
else:
    train_split = ['train', 'val']
    eval_split = 'test'
train_dict = make_train_input_tensors(dataset, dataset_name, train_split, save_file=False)
train_dict = extract_reallyheldout_by_id_partial(train_dict)
train_dict, fewshot_meta_data = fewshot_from_train(train_dict,Kvalues=Kvalues)
train_spikes_heldin = train_dict['train_spikes_heldin']
train_spikes_heldout = train_dict['train_spikes_heldout']
eval_dict = make_eval_input_tensors(dataset, dataset_name, eval_split, save_file=False)
eval_dict = extract_reallyheldout_by_id_partial(eval_dict)
eval_spikes_heldin = eval_dict['eval_spikes_heldin']

# ---- Convert to neo.SpikeTrains ---- #
def array_to_spiketrains(array, bin_size):
    """Convert B x T x N spiking array to list of list of SpikeTrains"""
    stList = []
    for trial in range(len(array)):
        trialList = []
        for channel in range(array.shape[2]):
            times = np.nonzero(array[trial, :, channel])[0]
            counts = array[trial, times, channel].astype(int)
            times = np.repeat(times, counts)
            st = neo.SpikeTrain(times*bin_size*pq.ms, t_stop=array.shape[1]*bin_size*pq.ms)
            trialList.append(st)
        stList.append(trialList)
    return stList

# fewshot_train_spikes_heldin = train_dict[f'{4}shot_id{0}_'+'train_spikes_heldin']
# fewshot_train_st_heldin = array_to_spiketrains(fewshot_train_spikes_heldin, bin_size_ms)
train_st_heldin = array_to_spiketrains(train_spikes_heldin, bin_size_ms)
eval_st_heldin = array_to_spiketrains(eval_spikes_heldin, bin_size_ms)

# ---- Run GPFA ---- #
gpfa = GPFA(bin_size=(bin_size_ms * pq.ms), x_dim=latent_dim) #, em_max_iters = 2)
train_factors = gpfa.fit_transform(train_st_heldin)
print('Evaluating GPFA model...')
eval_factors = gpfa.transform(eval_st_heldin)
# eval_factors_fewshot = gpfa.transform(fewshot_train_st_heldin)

# ---- Reshape factors ---- #
train_factors_s = np.vstack([train_factors[i].T for i in range(len(train_factors))])
eval_factors_s = np.vstack([eval_factors[i].T for i in range(len(eval_factors))])

# ---- Useful variables ---- #
hi_chan = train_spikes_heldin.shape[2]
ho_chan = train_spikes_heldout.shape[2]
tlength = train_spikes_heldin.shape[1]
num_train = len(train_st_heldin)
num_eval = len(eval_st_heldin)

# ---- Prepare data for regression ---- #
train_spikes_heldin_s = train_spikes_heldin.reshape(-1, train_spikes_heldin.shape[2])
train_spikes_heldout_s = train_spikes_heldout.reshape(-1, train_spikes_heldout.shape[2])
eval_spikes_heldin_s = eval_spikes_heldin.reshape(-1, eval_spikes_heldin.shape[2])

# ---- Define helpers ---- #
flatten2d = lambda x: x.reshape(-1, x.shape[2])

def fit_poisson(train_factors_s, eval_factors_s, train_spikes_s, eval_spikes_s=None, alpha=0.0):
    """Fit Poisson GLM from factors to spikes and return rate predictions"""
    train_in = train_factors_s if eval_spikes_s is None else np.vstack([train_factors_s, eval_factors_s])
    train_out = train_spikes_s if eval_spikes_s is None else np.vstack([train_spikes_s, eval_spikes_s])
    train_pred = []
    eval_pred = []
    for chan in range(train_out.shape[1]):
        pr = PoissonRegressor(alpha=alpha, max_iter=500)
        pr.fit(train_in, train_out[:, chan])
        while pr.n_iter_ == pr.max_iter and pr.max_iter < 10000:
            print(f"didn't converge - retraining {chan} with max_iter={pr.max_iter * 5}")
            oldmax = pr.max_iter
            del pr
            pr = PoissonRegressor(alpha=alpha, max_iter=oldmax * 5)
            pr.fit(train_in, train_out[:, chan])
        train_pred.append(pr.predict(train_factors_s))
        eval_pred.append(pr.predict(eval_factors_s))
    train_rates_s = np.vstack(train_pred).T
    eval_rates_s = np.vstack(eval_pred).T
    return train_rates_s, eval_rates_s

def fit_rectlin(train_factors_s, eval_factors_s, train_spikes_s, eval_spikes_s=None, alpha=0.0, thresh=1e-10):
    """Fit linear regression from factors to spikes, rectify, and return rate predictions"""
    train_in = train_factors_s if eval_spikes_s is None else np.vstack([train_factors_s, eval_factors_s])
    train_out = train_spikes_s if eval_spikes_s is None else np.vstack([train_spikes_s, eval_spikes_s])
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_in, train_out)
    train_rates_s = ridge.predict(train_factors_s)
    eval_rates_s = ridge.predict(eval_factors_s)
    rect_min = np.min([np.min(train_rates_s[train_rates_s > 0]), np.min(eval_rates_s[eval_rates_s > 0])])
    true_min = np.min([np.min(train_rates_s), np.min(eval_rates_s)])
    train_rates_s[train_rates_s < thresh] = thresh
    eval_rates_s[eval_rates_s < thresh] = thresh
    return train_rates_s, eval_rates_s, ridge

# ---- Rate prediction ---- #
train_rates_heldin_s, eval_rates_heldin_s, ridge = fit_rectlin(train_factors_s, eval_factors_s, train_spikes_heldin_s, eval_spikes_heldin_s, alpha=alpha1)
train_rates_heldout_s, eval_rates_heldout_s = fit_poisson(train_rates_heldin_s, eval_rates_heldin_s, train_spikes_heldout_s, alpha=alpha2)

train_rates_heldin = train_rates_heldin_s.reshape(num_train, tlength, hi_chan)
train_rates_heldout = train_rates_heldout_s.reshape(num_train, tlength, ho_chan)
eval_rates_heldin = eval_rates_heldin_s.reshape(num_eval, tlength, hi_chan)
eval_rates_heldout = eval_rates_heldout_s.reshape(num_eval, tlength, ho_chan)

# ----- Few shot rate predicion ---- #
# _, eval_rates_heldout_s = fit_poisson(train_rates_heldin_s, eval_rates_heldin_s, train_spikes_heldout_s, alpha=alpha2)
fewshot_output_dict = {}
# iterating over K values
for k in fewshot_meta_data['Kvalues_applicable']:
     # iterating over fewshot subsets
    for id in fewshot_meta_data[f'{k}shot_ids']:
        # print(f'{k}shot_id{id}_')
        fewshot_train_spikes_heldin = train_dict[f'{k}shot_id{id}_'+'train_spikes_heldin']
        fewshot_train_spikes_heldout = train_dict[f'{k}shot_id{id}_'+'train_spikes_heldout']
        # eval_spikes_heldin = eval_dict['eval_spikes_heldin'] # same evaluation set as the standard evaluation
        fewshot_train_spikes_heldout_s = fewshot_train_spikes_heldout.reshape(-1, fewshot_train_spikes_heldout.shape[2])
        
        fewshot_train_st_heldin = array_to_spiketrains(fewshot_train_spikes_heldin, bin_size_ms)
        fewshot_train_st_heldout = array_to_spiketrains(fewshot_train_spikes_heldout, bin_size_ms)
        # eval_st_heldout = array_to_spiketrains(eval_spikes_heldin, bin_size_ms)
        
        # print(fewshot_train_st_heldin.shape,fewshot_train_st_heldin.shape,eval_st_heldin.shape)
        fewshot_train_factors = gpfa.transform(fewshot_train_st_heldin)
        fewshot_train_factors_s = np.vstack([fewshot_train_factors[i].T for i in range(len(fewshot_train_factors))])
        # eval_factors = gpfa.transform(fewshot_train_st_heldin)

        fewshot_train_rates_s = ridge.predict(fewshot_train_factors_s)
        thresh = 1e-10
        fewshot_train_rates_s[fewshot_train_rates_s < thresh] = thresh
                
        (
            fewshot_train_rates_s, eval_rates_s
        ) = fit_poisson(
            fewshot_train_rates_s, eval_rates_heldin_s, fewshot_train_spikes_heldout_s, alpha=alpha2
        )
        eval_rates_heldout_fewshot = eval_rates_s.reshape(num_eval, tlength, ho_chan)
        # eval_rates_heldout_all.append(fewshot_eval_spksmth_heldout)
        fewshot_output_dict [f'{k}shot_id{id}_eval_rates_heldout'] = np.array(eval_rates_heldout_fewshot)


# ---- Save output ---- #
output_dict = {
    dataset_name + binsuf: {
        **fewshot_output_dict,
        'train_rates_heldin': train_rates_heldin,
        'train_rates_heldout': train_rates_heldout,
        'eval_rates_heldin': eval_rates_heldin,
        'eval_rates_heldout': eval_rates_heldout,
    }
}
save_to_h5(output_dict, savepath, overwrite=True)

if phase == 'val':
    target_dict = make_eval_target_tensors(dataset, dataset_name, train_split, eval_split, save_file=False, include_psth=True)
    for k,v in target_dict.items():
        target_dict[k] = extract_reallyheldout_by_id_partial(v)
    print(evaluate(target_dict, output_dict))
