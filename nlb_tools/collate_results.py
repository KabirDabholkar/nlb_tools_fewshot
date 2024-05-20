import os
import pandas as pd
from pathlib import Path
import hashlib


# def read_concat_results(root_dir:str,
#                         endswith:str='results.csv',
#                         include_path:bool = True):
#     # Initialize an empty DataFrame to store concatenated results
#     concatenated_results = pd.DataFrame()

#     # Iterate through the root directory and its subdirectories
#     for subdir, _, files in os.walk(root_dir):
#         # Check if any file named 'results.csv' exists in the current directory
#         for f in files:
#             if f.endswith(endswith) and 'concat' not in f:                    
#                 # Form the full path of the 'results.csv' file
#                 results_file_path = os.path.join(subdir, f)
                
#                 # Read the CSV file into a DataFrame
#                 df = pd.read_csv(results_file_path,index_col=0)
#                 if include_path:
#                     df['path'] = results_file_path
#                 # Concatenate the current DataFrame with the overall concatenated results
#                 concatenated_results = pd.concat([concatenated_results, df], ignore_index=True)

#     return concatenated_results

def generate_unique_prefix(path):
    return hashlib.md5(path.encode()).hexdigest()[:6]

def read_concat_results(root_dir: str,
                        endswith: str = 'results.csv',
                        include_path: bool = True):
    # Initialize an empty DataFrame to store concatenated results
    concatenated_results = pd.DataFrame()

    # Generate a unique prefix based on the root directory
    prefix = generate_unique_prefix(str(root_dir))

    # Initialize a counter for unique IDs
    model_id = 0

    # Create a dictionary to store the mapping from IDs to paths
    id_to_path = {}

    # Iterate through the root directory and its subdirectories
    for subdir, _, files in os.walk(root_dir):
        # Check if any file named 'results.csv' exists in the current directory
        for f in files:
            if f.endswith(endswith) and 'concat' not in f:                    
                # Form the full path of the 'results.csv' file
                results_file_path = os.path.join(subdir, f)
                suffix = generate_unique_prefix(results_file_path)


                # Read the CSV file into a DataFrame
                df = pd.read_csv(results_file_path, index_col=0)

                # Add a unique ID column
                df['id'] = f"{prefix}_{suffix}"
                
                # Optionally add the path
                if include_path:
                    df['path'] = results_file_path

                

                # Update the mapping dictionary
                id_to_path[f"{prefix}_{suffix}"] = results_file_path

                # Increment the model ID counter
                model_id += 1

                # Concatenate the current DataFrame with the overall concatenated results
                concatenated_results = pd.concat([concatenated_results, df], ignore_index=True)

    # Save the mapping dictionary for future reference if needed
    # mapping_df = pd.DataFrame(list(id_to_path.items()), columns=['id', 'path'])

    return concatenated_results #, mapping_df

# Example usage:
# root_dir = 'path/to/your/directory'
# concatenated_results, id_mapping = read_concat_results(root_dir)
# concatenated_results.to_csv('concatenated_results.csv', index=False)
# id_mapping.to_csv('id_to_path_mapping.csv', index=False)


def main():
    # Root directory containing 'results.csv' files and its subdirectories
    root_directory = Path('/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_dmfc_rsg/240501_030533_MultiFewshot')

    # Read and concatenate 'results.csv' files
    concatenated_results = read_concat_results(root_directory)

    # Save the concatenated results to a new CSV file
    concatenated_results.to_csv(root_directory / 'concatenated_results.csv')

    print("Concatenated results saved successfully!")

if __name__ == "__main__":
    main()
