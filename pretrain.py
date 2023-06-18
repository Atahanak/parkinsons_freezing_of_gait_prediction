#!/usr/bin/env python
# coding: utf-8

import os
import time
import random
import gc
import glob
import json

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
torch.set_float32_matmul_precision('high')

from watermark import watermark
print(watermark(packages="pytorch_lightning, torchmetrics, torch, sklearn, pandas, numpy"))

from config import Config
cfg = Config('pretrain-config')
print(vars(Config))

from models.models import FOGModel
from models.models import FOGTransformerEncoder
# get the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'FOGTransformerEncoder has {count_parameters(FOGTransformerEncoder(cfg, state="pretrain")):,} trainable parameters')
print(f'FOGModel has {count_parameters(FOGModel(cfg, state="pretrain")):,} trainable parameters')

from utils import *
from dataset.Dataset import FOGDataset
print("Initializing dataset...")
pretrain_paths = [(f, 'unlabeled') for f in glob.glob(f"{cfg['DATA_DIR']}unlabeled/*.parquet")]
fog_pretrain = FOGDataset(pretrain_paths, cfg, state="pretrain")
fog_train_loader = DataLoader(fog_pretrain, batch_size=cfg['batch_size']) #, num_workers=16) #cfg.num_workers)
print("Dataset size:", fog_pretrain.__len__())
print("Number of batches:", len(fog_train_loader))
print("Batch size:", fog_train_loader.batch_size)
print("Total size:", len(fog_train_loader) * fog_train_loader.batch_size)
gc.collect()

from modules.modules import FOGPreTrainModule

def pretrain_model(module, model, train_loader, save_name = None, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(cfg['CHECKPOINT_PATH'], save_name),                          # Where to save models
                         accelerator="gpu" if str(cfg['device']).startswith("cuda") else "cpu",                     # We run on a GPU (if possible)
                         devices=torch.cuda.device_count() if str(cfg['device']).startswith("cuda") else 1,                                                                          # How many GPUs/CPUs we want to use (1 is enough for the notebooks)
                         max_epochs=cfg['num_epochs'],                                                                     # How many epochs to train for if no patience is set
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="train_loss"),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                    LearningRateMonitor("epoch")],                                           # Log learning rate every epoch
                         enable_progress_bar=True,                                                          # Set to False if you do not want a progress bar
                         strategy=DDPStrategy(find_unused_parameters=True),
                         logger = True,
                         # val_check_interval=0.5,
                         log_every_n_steps=50)                                                           
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = True

    # log hyperparameters, including model and custom parameters
    model.hparams.update(cfg['hparams'])
    del model.hparams["milestones"] # = str(model.hparams["milestones"])
    trainer.logger.log_metrics(model.hparams)

    # Check whether pretrained model exists. If yes, load it and skip training
    pl.seed_everything(42) # To be reproducable
    lmodel = module(cfg, model, **kwargs)
    trainer.fit(lmodel, train_loader)
    lmodel = module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training
    
    #save model ready to be loaded for finetuning
    pretrained_file_name = os.path.join(cfg['CHECKPOINT_PATH'], save_name, f"{cfg['model_hidden']}_{cfg['model_nblocks']}_final.pt")
    print(f"Saving pretrained model to: {pretrained_file_name}")
    torch.save(lmodel.model.state_dict(), pretrained_file_name)

    train_loss = trainer.logged_metrics["train_loss"]
    result = {
        "train_loss": train_loss.item()
    }

    return lmodel, trainer, result


# In[16]:


print(f"# of devices: {torch.cuda.device_count()}")
model = FOGModel(cfg, state="pretrain")
model, trainer, result = pretrain_model(FOGPreTrainModule, model, fog_train_loader, save_name="FOGModel", optimizer_name="Adam", optimizer_hparams={"lr": cfg['lr'], "weight_decay": cfg['gamma']})
print(json.dumps(cfg['hparams'], sort_keys=True, indent=4))
print(json.dumps(result, sort_keys=True, indent=4))

"""
def train(model, optimizer, criterion, train_loader):
    print("Training...")
    for x, y, _ in tqdm(train_loader):
        # print(y)
        print(x.shape, y.shape)
        #ic(x, y)
        # single forward pass
        # cast x to the correct data type
        x = x.float()
        y_hat = model(x)
        print(y_hat)
        # print(soft(y_hat))
        # print(y_hat.shape)
        # print(y_hat.argmax(dim=-1))
        # calculate loss
        loss = criterion(y_hat, y)
        print(loss.item())
        # calculate gradients
        loss.backward()
        # update weights
        optimizer.step()
        print(y)
        break
model = FOGTransformerEncoder("pre-train")
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
criterion = nn.MSELoss()
soft = nn.Softmax(dim=-1)
train(model, optimizer, criterion, fog_train_loader)
"""

