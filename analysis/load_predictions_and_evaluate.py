# ---- Imports ---- #
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_fewshot import extract_reallyheldout_by_id, fewshot_from_train
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5, h5_to_dict
from nlb_tools.evaluation import evaluate
from functools import partial
import numpy as np
import h5py
import pandas as pd


output_paths = {
    'mc_maze_20': ['mc_maze_20_gpfa_output_val.h5','mc_maze_20_smoothing_output_val.h5'],
    'mc_maze_small_20': ['mc_maze_small_20_gpfa_output_val.h5','mc_maze_small_20_smoothing_output_val.h5']
}

# ---- Run Params ---- #
dataset_name = "mc_maze" # one of {'area2_bump', 'dmfc_rsg', 'mc_maze', 'mc_rtt', 
                            # 'mc_maze_large', 'mc_maze_medium', 'mc_maze_small'}
bin_size_ms = 20

# function to extract neurons for fewshot analysis
really_heldout_neurons_ids = np.arange(10,dtype=int)
extract_reallyheldout_by_id_partial = partial(extract_reallyheldout_by_id,neuron_ids_to_extract=really_heldout_neurons_ids)
Kvalues = (2**np.arange(2,15)).astype(int)
phase = 'val' 
log_offset = 1e-4 # amount to add before taking log to prevent log(0) error

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
savepath = f'{dataset_name}{"" if bin_size_ms == 5 else f"_{bin_size_ms}"}_smoothing_output_{phase}.h5'

# ---- Load data ---- #
dataset = NWBDataset(datapath, prefix, skip_fields=['hand_pos', 'cursor_pos', 'eye_pos', 'muscle_vel', 'muscle_len', 'joint_vel', 'joint_ang', 'force'])
dataset.resample(bin_size_ms)

# ---- Extract data ---- #
if phase == 'val':
    train_split = 'train'
    eval_split = 'val'
else:
    train_split = ['train', 'val']
    eval_split = 'test'


# train_dict = make_train_input_tensors(dataset, dataset_name, train_split, save_file=False, include_forward_pred=True)
# train_dict = extract_reallyheldout_by_id_partial(train_dict)
# train_dict, fewshot_meta_data = fewshot_from_train(train_dict,Kvalues=Kvalues)
# train_spikes_heldin = train_dict['train_spikes_heldin']
# train_spikes_heldout = train_dict['train_spikes_heldout']
# eval_dict = make_eval_input_tensors(dataset, dataset_name, eval_split, save_file=False)
# eval_dict = extract_reallyheldout_by_id_partial(eval_dict)
# eval_spikes_heldin = eval_dict['eval_spikes_heldin']


eval_report = []
dataset_key = f'{dataset_name}{"" if bin_size_ms == 5 else f"_{bin_size_ms}"}'
target_dict = make_eval_target_tensors(dataset, dataset_name, train_split, eval_split, save_file=False, include_psth=True)
for k,v in target_dict.items():
    target_dict[k] = extract_reallyheldout_by_id_partial(v)
for path in output_paths[dataset_key]:
    with h5py.File(path, 'r') as f:
        output_dict = h5_to_dict(f)


    result_data = evaluate(target_dict, output_dict)
   # Extracting the key dynamically
    key_name = list(result_data[0].keys())[0]

    # Extracting the keys (column names) and the values (data) from the dictionary
    columns = list(result_data[0][key_name].keys())
    values = list(result_data[0][key_name].values())

    # Splitting column names based on space ' '
    columns_modified = columns
    if any(['[' in c for c in columns]):
        columns_modified = [' '.join(col.split(' ')[1:]) for col in columns]

    # Creating a DataFrame
    df = pd.DataFrame([values], columns=columns_modified)

    # Adding the 'dataset' column
    dataset_name = key_name + '_' + columns[0].split(' ')[0][1:-1]
    df['dataset'] = dataset_name
    df['path'] = path
    eval_report.append(df)

D = pd.concat(eval_report,axis=0)
D.to_csv('results.csv')
    