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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import MixedPrecisionPlugin 
import torchmetrics
torch.set_float32_matmul_precision('high')

from watermark import watermark
print(watermark(packages="pytorch_lightning, torchmetrics, torch, sklearn, pandas, numpy"))

from utils import *


# In[165]:
from config import Config
cfg = Config(file_name='config')
#pretty print the config
print(json.dumps(cfg.data, indent=4, sort_keys=True))

#loss_weights = [0.5625255840823298, 0.4374744159176702]
loss_weights = [0.4374744159176702, 0.5625255840823298]
# total_td , event_total_td = event_analysis(cfg, 'tdcsfog')
# total_de , event_total_de = event_analysis(cfg, 'defog')
# total_no , event_total_no = event_analysis(cfg, 'notype')

# total = total_td + total_de + total_no
# event_total = event_total_td + event_total_de + event_total_no

# #total = 10251114
# #event_total = 6832879
# print(f"Total: {total}, number of events: {event_total}")

# #get loss weights using total no and event total
# loss_weights = [(total - event_total) / total, event_total / total ]

# print(f"Loss weights: {loss_weights}")

from utils import *
#split_analysis(cfg, 'tdcsfog')
train_fpaths_tdcs, valid_fpaths_tdcs = split_data(cfg, 'tdcsfog', 2)
#split_analysis(cfg, 'defog')
train_fpaths_de, valid_fpaths_de = split_data(cfg, 'defog', 2)
train_fpaths_no = glob.glob(f"{cfg['DATA_DIR']}/train/notype/*")
#train_fpaths = [(f, 'de') for f in train_fpaths_de] + [(f, 'tdcs') for f in train_fpaths_tdcs] 
train_fpaths = [(f, 'tdcs') for f in train_fpaths_tdcs] 
#train_fpaths = [(f, 'de') for f in train_fpaths_de] 
train_fpaths_2 = [(f, 'notype') for f in train_fpaths_no]
# train_fpaths = [(f, 'notype') for f in train_fpaths_no]
#valid_fpaths = [(f, 'de') for f in valid_fpaths_de] + [(f, 'tdcs') for f in valid_fpaths_tdcs]
valid_fpaths = [(f, 'tdcs') for f in valid_fpaths_tdcs] 
#valid_fpaths = [(f, 'de') for f in valid_fpaths_de] 

gc.collect()

from dataset.Dataset import FOGDataset
fog_train = FOGDataset(train_fpaths, cfg)
fog_train_2 = FOGDataset(train_fpaths_2, cfg)
fog_train_loader = DataLoader(fog_train, batch_size=cfg["batch_size"], shuffle=True, num_workers=16)
fog_train_loader_2 = DataLoader(fog_train_2, batch_size=cfg["batch_size"], shuffle=True, num_workers=16)
fog_valid = FOGDataset(valid_fpaths, cfg)
fog_valid_loader = DataLoader(fog_valid, batch_size=cfg["batch_size"], num_workers=16)

# load no label dataset

print("TRAIN")
print("Dataset size:", fog_train.__len__())
print("Number of batches:", len(fog_train_loader))
print("Batch size:", fog_train_loader.batch_size)
print("Total size:", len(fog_train_loader) * fog_train_loader.batch_size)
print("Dataset size:", fog_train.__len__())
print("Number of batches:", len(fog_train_loader_2))
print("Batch size:", fog_train_loader_2.batch_size)
print("Total size:", len(fog_train_loader_2) * fog_train_loader_2.batch_size)
print("VALID")
print("Dataset size:", fog_valid.__len__())
print("Number of batches:", len(fog_valid_loader))
print("Batch size:", fog_valid_loader.batch_size)
print("Total size:", len(fog_valid_loader) * fog_valid_loader.batch_size)

from models.models import FOGCNNEventSeperator
model = FOGCNNEventSeperator(cfg)
#y = model(next(iter(fog_train_loader))[0].float())
print(f'The model has {count_parameters(model):,} trainable parameters')

from modules.modules import FOGEventSeperatorModule
from modules.callbacks import ConfusionMatrixCallback
from pytorch_lightning.utilities import CombinedLoader

def train_model(module, model, train_loaders, val_loader, test_loader, save_name = None, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    # Create a PyTorch Lightning trainer with the generation callback
    conf_matrix_callback = ConfusionMatrixCallback(cfg, num_classes=2, task="multiclass")
    combined_train_loaders = CombinedLoader(train_loaders, mode="sequential")
    trainer = pl.Trainer(default_root_dir=os.path.join(cfg['CHECKPOINT_PATH'], save_name),                          # Where to save models
                         plugins=[MixedPrecisionPlugin(precision="bf16-mixed", device=cfg['device'])],
                         accelerator="gpu" if str(cfg['device']).startswith("cuda") else "cpu",                     # We run on a GPU (if possible)
                         devices=torch.cuda.device_count() if str(cfg['device']).startswith("cuda") else 1,         # How many GPUs/CPUs we want to use (1 is enough for the notebooks)
                         max_epochs=cfg['num_epochs'],                                                                     # How many epochs to train for if no patience is set
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="avg_val_precision"),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                          conf_matrix_callback,
                                    LearningRateMonitor("epoch")],                                           # Log learning rate every epoch
                         enable_progress_bar=True,                                                          # Set to False if you do not want a progress bar
                         logger = True,
                         strategy=DDPStrategy(find_unused_parameters=True),
                         #val_check_interval=0.5,
                         log_every_n_steps=50)                                                           
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = True

    # log hyperparameters, including model and custom parameters
    del cfg['hparams']["milestones"] 
    trainer.logger.log_metrics(cfg["hparams"])

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(cfg["CHECKPOINT_PATH"], save_name + f"/{cfg['model_hidden']}_{cfg['model_nblocks']}_final.pt")
    print(f"Pretrained file {pretrained_filename}")
    if cfg['hparams']["use_pretrained"] and os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model.load_state_dict(torch.load(pretrained_filename)) # Automatically loads the model with the saved hyperparameters
    lmodel = module(cfg, model, loss_weights, **kwargs)

    # tune learning rate
    print("Tuning learning rate...")
    tuner = Tuner(trainer)
    # Run learning rate finder
    lr_finder = tuner.lr_find(lmodel, train_dataloaders=train_loaders[0], val_dataloaders=val_loader)
    # Auto-scale batch size with binary search
    #tuner.scale_batch_size(lmodel, mode="binsearch")
    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    print(f"New learning rate: {new_lr}")
    print("Tuning done.")
    
    pl.seed_everything(42) # To be reproducable
    trainer.fit(lmodel, train_loaders[0], val_loader)
    #trainer.fit(lmodel, train_loaders[1], val_loader)
    print(f"Best model path {trainer.checkpoint_callback.best_model_path}")
    lmodel = module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on validation set
    val_result = trainer.test(lmodel, val_loader, verbose=False)
    result = {
                "val_ap": val_result[0]["avg_val_precision"],
                "no": val_result[0]["val0_precision"],
                "yes": val_result[0]["val1_precision"],
            }

    return lmodel, trainer, result

model, trainer, result = train_model(FOGEventSeperatorModule, model, [fog_train_loader, fog_train_loader_2], fog_valid_loader, fog_valid_loader, save_name="FOGEventSeperator", optimizer_name="Adam")
print(json.dumps(cfg['hparams'], sort_keys=True, indent=4))
print(json.dumps(result, sort_keys=True, indent=4))

