import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

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

if __name__ == "__main__":
    plot_cross_decoding_scores(
        path_to_scores_numpy = '/home/kabird/STNDT_fewshot/ray_results/mc_maze_20_lite_all4_cross_decoding_scores.npy',
        path_to_savefig = 'plots/cross_decoding_STNDT_mc_maze_20.png' 
    )

