import pandas as pd


# path1 = '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240518_134433_MultiFewshot/concatenated_results_with_kshot.csv'
# path2 = '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240518_134433_MultiFewshot/concatenated_results.csv'
# new_path = '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240518_134433_MultiFewshot/concatenated_with_kshot_and_path_to_latents.csv'

# D1 = pd.read_csv(path1,index_col=0)
# D2 = pd.read_csv(path2,index_col=0)

# D3 = D1
# D3['path'] = D2['path']

# D3.to_csv(new_path)


path1 = '/home/kabird/STNDT_fewshot/ray_results/mc_maze_20_lite_all9.csv'
path2 = '/home/kabird/STNDT_fewshot/ray_results/mc_maze_20_lite_all10.csv'
new_path = '/home/kabird/STNDT_fewshot/ray_results/mc_maze_20_lite_all9and10.csv'

D1 = pd.read_csv(path1,index_col=0)
D2 = pd.read_csv(path2,index_col=0)

D3 = pd.concat([D1,D2],axis=0).drop_duplicates(subset=['id'])
D3.to_csv(new_path)
