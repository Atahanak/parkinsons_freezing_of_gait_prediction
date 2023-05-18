#!/usr/bin/env python
# coding: utf-8

# In[161]:


import os
import time
import random
import gc
import glob
import json


# In[162]:


from tqdm.notebook import tqdm
import numpy as np
import pandas as pd


# In[163]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.plugins import MixedPrecisionPlugin 
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchmetrics.functional.classification import multiclass_average_precision
import torchmetrics

torch.set_float32_matmul_precision('high')

# In[164]:

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import average_precision_score

# In[165]:
from config import Config
cfg = Config()
print(vars(Config))

"""
# Analysis of positive instances in each fold of our CV folds

SH = []
T = []
W = []

# Here I am using the metadata file available during training. Since the code will run again during submission, if 
# I used the usual file from the competition folder, it would have been updated with the test files too.
metadata = pd.read_csv(f"{cfg.DATA_DIR}tdcsfog_metadata.csv")

for f in tqdm(metadata['Id']):
    fpath = f"{cfg.TRAIN_DIR}tdcsfog/{f}.csv"
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
"""


# In[169]:


# The actual train-test split (based on Fold 2)

metadata = pd.read_csv(f"{cfg.DATA_DIR}tdcsfog_metadata.csv")
sgkf = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
for i, (train_index, valid_index) in enumerate(sgkf.split(X=metadata['Id'], y=[1]*len(metadata), groups=metadata['Subject'])):
    if i != 2:
        continue
    print(f"Fold = {i}")
    train_ids = metadata.loc[train_index, 'Id']
    valid_ids = metadata.loc[valid_index, 'Id']
    print(f"Length of Train = {len(train_ids)}, Length of Valid = {len(valid_ids)}")
    
    if i == 2:
        break
        
train_fpaths_tdcs = [f"{cfg.DATA_DIR}train/tdcsfog/{_id}.csv" for _id in train_ids if os.path.exists(f"{cfg.DATA_DIR}train/tdcsfog/{_id}.csv")]
valid_fpaths_tdcs = [f"{cfg.DATA_DIR}train/tdcsfog/{_id}.csv" for _id in valid_ids if os.path.exists(f"{cfg.DATA_DIR}train/tdcsfog/{_id}.csv")]


metadata = pd.read_csv(f"{cfg.DATA_DIR}defog_metadata.csv")
sgkf = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
for i, (train_index, valid_index) in enumerate(sgkf.split(X=metadata['Id'], y=[1]*len(metadata), groups=metadata['Subject'])):
    if i != 2:
        continue
    print(f"Fold = {i}")
    train_ids = metadata.loc[train_index, 'Id']
    valid_ids = metadata.loc[valid_index, 'Id']
    print(f"Length of Train = {len(train_ids)}, Length of Valid = {len(valid_ids)}")
    
    if i == 2:
        break
        
train_fpaths_de = [f"{cfg.DATA_DIR}train/defog/{_id}.csv" for _id in train_ids if os.path.exists(f"{cfg.DATA_DIR}train/defog/{_id}.csv")]
valid_fpaths_de = [f"{cfg.DATA_DIR}train/defog/{_id}.csv" for _id in valid_ids if os.path.exists(f"{cfg.DATA_DIR}train/defog/{_id}.csv")]

train_fpaths = [(f, 'de') for f in train_fpaths_de] + [(f, 'tdcs') for f in train_fpaths_tdcs]
valid_fpaths = [(f, 'de') for f in valid_fpaths_de] + [(f, 'tdcs') for f in valid_fpaths_tdcs]


# In[170]:


gc.collect()


# In[171]:
from dataset.Dataset import FOGDataset

fog_train = FOGDataset(train_fpaths, cfg)
fog_train_loader = DataLoader(fog_train, batch_size=cfg.batch_size, shuffle=True, num_workers=16)


# In[172]:


fog_valid = FOGDataset(valid_fpaths, cfg)
fog_valid_loader = DataLoader(fog_valid, batch_size=cfg.batch_size, num_workers=16)

# In[173]:


print("Dataset size:", fog_train.__len__())
print("Number of batches:", len(fog_train_loader))
print("Batch size:", fog_train_loader.batch_size)
print("Total size:", len(fog_train_loader) * fog_train_loader.batch_size)

# In[174]:

print("Dataset size:", fog_valid.__len__())
print("Number of batches:", len(fog_valid_loader))
print("Batch size:", fog_valid_loader.batch_size)
print("Total size:", len(fog_valid_loader) * fog_valid_loader.batch_size)


# In[190]:
from models.models import FOGModel
from models.models import FOGTransformerEncoder


# get the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(FOGTransformerEncoder(cfg)):,} trainable parameters')
print(f'The model has {count_parameters(FOGModel(cfg)):,} trainable parameters')

from modules.modules import FOGModule
from modules.callbacks import ConfusionMatrixCallback

# open events.csv file read it and store it in a dataframe
loss_weights = []
with open(f"{cfg.DATA_DIR}/events.csv") as file:
    df = pd.read_csv(file)
    # create a new dataframe column by take the difference between begin and end columns
    df['duration'] = df['Completion'] - df['Init']
    # take the mean of the duration column
    mean = df['duration'].mean()
    print(df)
    # create a new dataframe by grouping each task according to their mean duration and also count 
    df = df.groupby('Type').agg({'duration': ['sum', 'count']})
    # sort dt by mean duration
    df = df.sort_values(by=[('duration', 'sum')], ascending=False)
    # get numpy array of duration sum inversely scaled between 0 and 1
    counts = df[('duration', 'sum')].values
    loss_weights = 1 - (counts / counts.sum()) # inverse scale
    #loss_weights = 1 / counts # inverse scale
    #loss_weights = loss_weights / loss_weights.sum() * 3
    print("Loss weights: ", loss_weights, loss_weights.sum())

def train_model(module, model, train_loader, val_loader, test_loader, save_name = None, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    # Create a PyTorch Lightning trainer with the generation callback
    conf_matrix_callback = ConfusionMatrixCallback(cfg, num_classes=3, task="multiclass")
    trainer = pl.Trainer(default_root_dir=os.path.join(cfg.CHECKPOINT_PATH, save_name),                          # Where to save models
                         plugins=[MixedPrecisionPlugin(precision="bf16-mixed", device="cuda")],
                         accelerator="gpu" if str(cfg.device).startswith("cuda") else "cpu",                     # We run on a GPU (if possible)
                         devices=torch.cuda.device_count() if str(cfg.device).startswith("cuda") else 1,         # How many GPUs/CPUs we want to use (1 is enough for the notebooks)
                         max_epochs=cfg.num_epochs,                                                                     # How many epochs to train for if no patience is set
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="avg_val_precision"),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                    conf_matrix_callback,
                                    LearningRateMonitor("epoch")],                                           # Log learning rate every epoch
                         enable_progress_bar=True,                                                          # Set to False if you do not want a progress bar
                         logger = True,
                         strategy=DDPStrategy(find_unused_parameters=True),
                         val_check_interval=0.5,
                         log_every_n_steps=50)                                                           
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = True

    # log hyperparameters, including model and custom parameters
    model.hparams.update(cfg.hparams)
    del model.hparams["milestones"] 
    trainer.logger.log_metrics(model.hparams)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(cfg.CHECKPOINT_PATH, save_name + f"/{cfg.model_hidden}_{cfg.model_nblocks}_final.pt")
    print(f"Pretrained file {pretrained_filename}")
    if cfg.hparams["use_pretrained"] and os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model.load_state_dict(torch.load(pretrained_filename)) # Automatically loads the model with the saved hyperparameters
    lmodel = module(cfg, model, loss_weights, **kwargs)

    # tune learning rate
    print("Tuning learning rate...")
    tuner = Tuner(trainer)
    # Run learning rate finder
    lr_finder = tuner.lr_find(lmodel, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    print(f"New learning rate: {new_lr}")
    print("Tuning done.")
    
    pl.seed_everything(42) # To be reproducable
    trainer.fit(lmodel, train_loader, val_loader)
    print(f"Best model path {trainer.checkpoint_callback.best_model_path}")
    lmodel = module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on validation set
    val_result = trainer.test(lmodel, val_loader, verbose=False)
    result = {
                "val_ap": val_result[0]["avg_val_precision"],
                "SH": val_result[0]["val0_precision"],
                "T": val_result[0]["val1_precision"],
                "W": val_result[0]["val2_precision"]
            }

    return lmodel, trainer, result


# In[193]:


model = FOGModel(cfg)
model, trainer, result = train_model(FOGModule, model, fog_train_loader, fog_valid_loader, fog_valid_loader, save_name="FOGModel", optimizer_name="Adam")
print(json.dumps(cfg.hparams, sort_keys=True, indent=4))
print(json.dumps(result, sort_keys=True, indent=4))


# ## Submission

model = FOGModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
model.to(cfg.device)
model.eval()

test_defog_paths = glob.glob(f"{cfg.DATA_DIR}test/defog/*.csv")
test_tdcsfog_paths = glob.glob(f"{cfg.DATA_DIR}test/tdcsfog/*.csv")
test_fpaths = [(f, 'de') for f in test_defog_paths] + [(f, 'tdcs') for f in test_tdcsfog_paths]

test_dataset = FOGDataset(test_fpaths, split="test")
test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

ids = []
preds = []

for _id, x, _ in tqdm(test_loader):
    x = x.to(cfg.device).float()
    with torch.no_grad():
        y_pred = model(x)*0.1

    ids.extend(_id)
    preds.extend(list(np.nan_to_num(y_pred.cpu().numpy())))

sample_submission = pd.read_csv(f"{cfg.DATA_DIR}sample_submission.csv")
print(sample_submission.shape)

preds = np.array(preds)
submission = pd.DataFrame({'Id': ids, 'StartHesitation': np.round(preds[:,0],5), \
                           'Turn': np.round(preds[:,1],5), 'Walking': np.round(preds[:,2],5)})

submission = pd.merge(sample_submission[['Id']], submission, how='left', on='Id').fillna(0.0)
submission.to_csv(f"submission.csv", index=False)

print(submission.shape)
submission.head()

