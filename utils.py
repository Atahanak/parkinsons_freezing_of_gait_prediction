from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import os
import tqdm
import numpy as np

def split_data(cfg, folder_name, split=2):
    metadata = pd.read_csv(f"{cfg['DATA_DIR']}{folder_name}_metadata.csv")
    sgkf = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    for i, (train_index, valid_index) in enumerate(sgkf.split(X=metadata['Id'], y=[1]*len(metadata), groups=metadata['Subject'])):
        if i != split:
            continue
        print(f"Fold = {i}")
        train_ids = metadata.loc[train_index, 'Id']
        valid_ids = metadata.loc[valid_index, 'Id']
        print(f"Length of Train = {len(train_ids)}, Length of Valid = {len(valid_ids)}")
        
        if i == split:
            break
            
    train_fpaths_tdcs = [f"{cfg['DATA_DIR']}train/{folder_name}/{_id}.csv" for _id in train_ids if os.path.exists(f"{cfg['DATA_DIR']}train/{folder_name}/{_id}.csv")]
    valid_fpaths_tdcs = [f"{cfg['DATA_DIR']}train/{folder_name}/{_id}.csv" for _id in valid_ids if os.path.exists(f"{cfg['DATA_DIR']}train/{folder_name}/{_id}.csv")]
    return train_fpaths_tdcs, valid_fpaths_tdcs

def split_analysis(cfg, folder_name):
    # Analysis of positive instances in each fold of our CV folds

    SH = []
    T = []
    W = []

    # Here I am using the metadata file available during training. Since the code will run again during submission, if 
    # I used the usual file from the competition folder, it would have been updated with the test files too.
    metadata = pd.read_csv(f"{cfg['DATA_DIR']}{folder_name}_metadata.csv")

    for f in tqdm(metadata['Id']):
        fpath = f"{cfg['TRAIN_DIR']}{folder_name}/{f}.csv"
        df = pd.read_csv(fpath)
        
        SH.append(np.sum(df['StartHesitation']))
        T.append(np.sum(df['Turn']))
        W.append(np.sum(df['Walking']))

    metadata['SH'] = SH
    metadata['T'] = T
    metadata['W'] = W

    sgkf = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    for i, (train_index, valid_index) in enumerate(sgkf.split(X=metadata['Id'], y=[1]*len(metadata), groups=metadata['Subject'])):
        print(f"Fold = {i}")
        train_ids = metadata.loc[train_index, 'Id']
        valid_ids = metadata.loc[valid_index, 'Id']
        
        print(f"Length of Train = {len(train_ids)}, Length of trainid = {len(valid_index)}")
        n1_sum = metadata.loc[train_index, 'SH'].sum()
        n2_sum = metadata.loc[train_index, 'T'].sum()
        n3_sum = metadata.loc[train_index, 'W'].sum()
        print(f"Train classes: {n1_sum:,}, {n2_sum:,}, {n3_sum:,}")
        
        n1_sum = metadata.loc[valid_index, 'SH'].sum()
        n2_sum = metadata.loc[valid_index, 'T'].sum()
        n3_sum = metadata.loc[valid_index, 'W'].sum()
        print(f"Valid classes: {n1_sum:,}, {n2_sum:,}, {n3_sum:,}")
        
    # # FOLD 2 is the most well balanced

import glob
def event_analysis(cfg, folder_name):
    files = glob.glob(f"{cfg['TRAIN_DIR']}{folder_name}/*.csv")
    events = []
    for f in files:
        df = pd.read_csv(f)
        # get the number of rows in the dataframe
        rows = len(df.index)
        # produce a list of sum of the columns 5, 6, 7
        event = df.iloc[:, 5:8].sum().values
        events.append((rows, sum(event)))
    total = np.array(events).sum(axis=0)
    print(f"Total {folder_name} events: {total}")
    #convert to tuple
    return total


# get the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)