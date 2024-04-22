from itertools import product
import numpy as np
import pandas as pd
import h5py
import sys
import os
from typing import Optional,Iterable


def fewshot_from_train(
        train_dict: dict,
        Kvalues : Iterable[int] = []):
    num_trials_all = [t.shape[0] for t in train_dict.values()]
    num_trials = num_trials_all[0]
    assert all([t == num_trials for t in num_trials_all])
    Kvalues = np.array(Kvalues)
    Kvalues_applicable = Kvalues[Kvalues<num_trials]

    updates = {}
    meta_data = {}
    meta_data['Kvalues_applicable'] = [int(i) for i in list(Kvalues_applicable)]
    for key,val in train_dict.items():
        for K in Kvalues_applicable:
            chunks = split_into_chunks_of_K(val, K)
            meta_data[f'{K}shot_ids'] = list(range(len(chunks)))
            for id,chunk in enumerate(chunks):
                updates[f'{K}shot_id{id}_'+key] = chunk
            
    train_dict = {
        **train_dict,
        **updates
    }
    return train_dict,meta_data

def split_into_chunks_of_K(train_array,K):
    max_len = (train_array.shape[0]//K) * K
    return np.split(train_array[:max_len],np.arange(K, max_len, K),axis=0)


def extract_reallyheldout_by_id(data_dict,
                                neuron_ids_to_extract: Iterable[int] = []):
    """
    Split the heldin neurons to keep some for few shot evaluation.
    Call this set `neuron_ids_to_extract`. 
    i.e split heldin neurons into [new_heldin,reallyheldout].
    
    Modify psth tensors if they exist, removing `reallyheldout` neurons
    in order to fit the shape [new_heldin,heldout].
    """
    
    updates_dict = {}
    for k, v in data_dict.items():
        if "heldin" in k:
            heldin_key = k
            reallyheldout_key = k.replace("heldin", "reallyheldout")
            heldin_neurons = v
            
            neurons_to_extract_mask = np.isin(np.arange(heldin_neurons.shape[-1],dtype=int),neuron_ids_to_extract) #np.array([np.isin(c,neuron_ids_to_extract) for c in range(heldin_neurons)],dtype=bool)
            
            reallyheldout_neurons = heldin_neurons[..., neurons_to_extract_mask]
            new_heldin_neurons    = heldin_neurons[...,~neurons_to_extract_mask]
            
            
            updates_dict[heldin_key] = new_heldin_neurons
            updates_dict[reallyheldout_key] = reallyheldout_neurons
    
    if 'psth' in data_dict.keys():
        psth = data_dict['psth']
        neurons_to_extract_mask = np.isin(np.arange(psth.shape[-1],dtype=int),neuron_ids_to_extract)
        reallyheldout_psth = psth[..., neurons_to_extract_mask]
        new_psth           = psth[...,~neurons_to_extract_mask]

        updates_dict['psth'] = new_psth
        updates_dict['realy_heldout_psth'] = reallyheldout_psth
    data_dict.update(updates_dict)

    return data_dict


def split_heldin(
        heldin_neurons,
        num_reallyheldout_neurons: Optional[int] = None,
        split_keep_first: bool = True
    ):
    order = 1 if split_keep_first else -1
    reallyheldout_neurons, new_heldin_neurons  = np.split(
        heldin_neurons[...,::order], [num_reallyheldout_neurons], axis=-1
    )
    new_heldin_neurons, reallyheldout_neurons = (
        new_heldin_neurons[...,::order],
        reallyheldout_neurons[...,::order],
    )
    return new_heldin_neurons,reallyheldout_neurons

def extract_reallyheldout(data_dict,
                          num_reallyheldout_neurons: Optional[int] = None,
                          split_keep_first: bool = True):
    """
    Split the heldin neurons to keep some for few shot evaluation.
    Call this set `reallyheldout`. 
    i.e split heldin neurons into [new_heldin,reallyheldout].
    
    Modify psth tensors if they exist, removing `reallyheldout` neurons
    in order to fit the shape [new_heldin,heldout].
    """
    # base_heldin_key = [k for k in data_dict.keys() if 'heldin' in k]

    updates_dict = {}
    for k, v in data_dict.items():
        if "heldin" in k:
            heldin_key = k
            reallyheldout_key = k.replace("heldin", "reallyheldout")
            heldin_neurons = v
            
            num_heldin_neurons = v.shape[2]
            if num_reallyheldout_neurons is None:
                num_reallyheldout_neurons = num_heldin_neurons // 4
            
            # encod_neurons, heldout_neurons = np.split(v, [num_heldin_neurons], axis=2)
            # heldout_neurons = heldout_neurons[...,np.random.permutation(heldout_neurons.shape[-1])]

            # order = 1 if split_keep_first else -1
            # reallyheldout_neurons, new_heldin_neurons  = np.split(
            #     heldin_neurons[...,::order], [num_reallyheldout_neurons], axis=2
            # )
            # new_heldin_neurons, reallyheldout_neurons = (
            #     new_heldin_neurons[...,::order],
            #     reallyheldout_neurons[...,::order],
            # )
            new_heldin_neurons,reallyheldout_neurons = split_heldin(
                heldin_neurons=heldin_neurons,
                num_reallyheldout_neurons=num_reallyheldout_neurons,
                split_keep_first=split_keep_first
            )
            updates_dict[heldin_key] = new_heldin_neurons
            updates_dict[reallyheldout_key] = reallyheldout_neurons
    # if 'psth' in data_dict.keys():
    #     num_new_recon_neurons = updates_dict['train_recon_data'].shape[-1] #num_recon_neurons - num_reallyheldout_neurons
    #     heldin_psth,heldout_psth = np.split(data_dict['psth'],indices_or_sections=,axis=-1)
    #     # psth_dict = {
    #     #     'psth_heldin':heldin_psth,
    #     #     'psth_heldout':heldout_psth
    #     # }
    #     k = 'psth_heldin'
    #     reallyheldout_key = k.replace("heldin", "reallyheldout")
    #     v = heldin 
        
    #     num_heldin_neurons = v.shape[2]
    #     if num_reallyheldout_neurons is None:
    #         num_reallyheldout_neurons = num_heldin_neurons // 4
        
    #     # encod_neurons, heldout_neurons = np.split(v, [num_heldin_neurons], axis=2)
    #     # heldout_neurons = heldout_neurons[...,np.random.permutation(heldout_neurons.shape[-1])]

    #     new_heldin_neurons,reallyheldout_neurons = split_heldin(
    #         heldin_neurons=heldin_psth,
    #         num_reallyheldout_neurons=num_reallyheldout_neurons,
    #         split_keep_first=split_keep_first
    #     )

    #     updates_dict['psth'] = [...,:num_new_recon_neurons]
    data_dict.update(updates_dict)

    return data_dict

if __name__ == '__main__':
    print(
        split_into_chunks_of_K(np.arange(123),5)
    )