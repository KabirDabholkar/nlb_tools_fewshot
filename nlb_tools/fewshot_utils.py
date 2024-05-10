import pandas as pd

def result_dict_to_pandas(result_data,**kwargs):
    if len(result_data)==0:
        return pd.DataFrame([]) 
    key_name = list(result_data[0].keys())[0]

    # Extracting the keys (column names) and the values (data) from the dictionary
    columns = list(result_data[0][key_name].keys())
    values = list(result_data[0][key_name].values())

    # Splitting column names based on space ' '
    columns_modified = columns
    there_is_info_about_split = any(['[' in c for c in columns])
    if there_is_info_about_split:
        columns_modified = [' '.join(col.split(' ')[1:]) for col in columns]

    # Creating a DataFrame
    df = pd.DataFrame([values], columns=columns_modified)

    # Adding the 'dataset' column
    dataset_name = key_name + ('_' + columns[0].split(' ')[0][1:-1] if there_is_info_about_split else '')
    df['dataset'] = dataset_name
    for key,val in kwargs.items():
        df[key] = val
    return df