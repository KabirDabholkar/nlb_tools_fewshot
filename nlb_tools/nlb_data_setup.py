# Author: Trung Le
# Set up train, val, test split from NLB NWB data
# Curated from `basic_example.ipynb` in `nlb_tools` available at https://github.com/neurallatents/nlb_tools/blob/main/examples/tutorials/basic_example.ipynb

import numpy as np
import pandas as pd
import h5py
import argparse
import logging
logging.basicConfig(level=logging.INFO)

from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5, combine_h5
from nlb_tools.evaluation import evaluate

# # If necessary, download datasets from DANDI and put them in './data/'
# !pip install dandi
# !dandi download https://dandiarchive.org/dandiset/000128 # replace URL with URL for dataset you want
# # URLS are:
# # - MC_Maze: https://dandiarchive.org/dandiset/000128
# # - MC_RTT: https://dandiarchive.org/dandiset/000129
# # - Area2_Bump: https://dandiarchive.org/dandiset/000127
# # - DMFC_RSG: https://dandiarchive.org/dandiset/000130
# # - MC_Maze_Large: https://dandiarchive.org/dandiset/000138
# # - MC_Maze_Medium: https://dandiarchive.org/dandiset/000139
# # - MC_Maze_Small: https://dandiarchive.org/dandiset/000140

def get_parser():
    r"""
    Gets parsed arguments from command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset-name",
        choices=["mc_maze", "mc_maze_large", "mc_maze_medium", "mc_maze_small", "mc_rtt", "area2_bump", "dmfc_rsg"],
        help="name of dataset to generate training and evaluation data",
    )
    parser.add_argument(
        "bin-width",
        choices=['5','20'],
        help="Bin width in milliseconds. Either 5 or 20.",
    )
    parser.add_argument(
        "num-really-heldout-neurons",
        type=int,
        help="Number of data channels (neurons) to set aside for few-shot generalisation."
    )
    return parser


def main():
    r"""
    Sets up train, val, test splits from NWB data files
    Saves processed data splits to ./data/ in h5 format
    """
    parser = get_parser()
    args = parser.parse_args()
    dataset_name = vars(args)['dataset-name']
    bin_width = int(vars(args)['bin-width'])
    data_path_base = '/home/kabird/datasets'
    datapath_dict = {'mc_maze': f'{data_path_base}/000128/sub-Jenkins/',
                     'mc_maze_large': f'{data_path_base}/000138/sub-Jenkins/',
                     'mc_maze_medium': f'{data_path_base}/000139/sub-Jenkins/',
                     'mc_maze_small': f'{data_path_base}/000140/sub-Jenkins/',
                     'mc_rtt': f'{data_path_base}/000129/sub-Indy/',
                     'area2_bump': f'{data_path_base}/000127/sub-Han/',
                     'dmfc_rsg': f'{data_path_base}/000130/sub-Haydn/',
                     }
    datapath = datapath_dict[f'{dataset_name}']
    ## Load data from NWB file:
    dataset = NWBDataset(datapath)

    num_reallyheldout = int(vars(args)['num-really-heldout-neurons'])
    really_heldout_ids = np.arange(0,num_reallyheldout)
    
    
    dataset.resample(bin_width)
    K_values = (2**np.arange(2,15)).astype(int)

    # Create suffix for group naming later
    suffix = '' if (bin_width == 5) else f'_{int(bin_width)}'
    dataset_name_suffix = dataset_name + suffix
    print('dataset_name_suffix:',dataset_name_suffix)
    # Choose the phase here, either 'val' for the Validation phase or 'test' for the Test phase
    # Note terminology overlap with 'train', 'val', and 'test' data splits -
    # the phase name corresponds to the data split that predictions are evaluated on
    for phase in ['val', 'test']:
        ## Make train data
        # Create input tensors, returned in dict form
        train_split = 'train' if (phase == 'val') else ['train', 'val']
        train_dict = make_train_input_tensors(dataset, dataset_name=dataset_name, trial_split=train_split, 
                                                include_behavior=True, include_forward_pred=True, save_file=True,
                                                save_path=f"{data_path_base}/{dataset_name_suffix}_{'train' if phase=='val' else 'trainval'}.h5",
                                                really_heldout_ids=really_heldout_ids,
                                                fewshot_Kvalues=K_values)

        ## Make eval data
        # Split for evaluation is same as phase name
        eval_split = phase
        # Make data tensors
        eval_dict = make_eval_input_tensors(dataset, dataset_name=dataset_name, trial_split=eval_split, save_file=True,
                                            save_path=f"{data_path_base}/{dataset_name_suffix}_{phase}.h5")

        combine_h5([f"{data_path_base}/{dataset_name_suffix}_{'train' if phase=='val' else 'trainval'}.h5", f"{data_path_base}/{dataset_name_suffix}_{phase}.h5"], save_path=f"{data_path_base}/{dataset_name_suffix}{'' if phase=='val' else '_test'}_full.h5")

    ## Make data to evaluate predictions with
    # Reset logging level to hide excessive info messages
    logging.getLogger().setLevel(logging.WARNING)

    phase = 'val'
    # If 'val' phase, make the target data
    if phase == 'val':
        # Note that the RTT task is not well suited to trial averaging, so PSTHs are not made for it
        target_dict = make_eval_target_tensors(dataset, dataset_name=dataset_name, train_trial_split='train', eval_trial_split='val', 
                                                include_psth=True, save_file=True, save_path=f"{data_path_base}/{dataset_name_suffix}_target.h5")


if __name__ == "__main__":
    main()

