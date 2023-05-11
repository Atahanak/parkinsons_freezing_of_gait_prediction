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
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchmetrics.functional.classification import multiclass_average_precision
import torchmetrics

torch.set_float32_matmul_precision('high')

# In[164]:


from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import average_precision_score

from positional_encoding import *

from watermark import watermark
print(watermark(packages="pytorch_lightning, torchmetrics, torch, sklearn, pandas, numpy"))


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

for f in tqdm(metadata['Id']):3421113
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
gc.collect()

from Dataset import FOGPatchDataSet
fog_train = FOGPatchDataSet(train_fpaths, cfg)
fog_train_loader = DataLoader(fog_train, batch_size=cfg.batch_size, shuffle=True) #, num_workers=16)


# In[172]:


fog_valid = FOGPatchDataSet(valid_fpaths, cfg)
fog_valid_loader = DataLoader(fog_valid, batch_size=cfg.batch_size) #, num_workers=16)

# x = next(iter(fog_train_loader))
# print(x[0].shape, x[1].shape)

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

from src.models.patchTST import PatchTST

# In[190]:
model = PatchTST(int(cfg.num_vars), int(cfg.num_classes), int(cfg.patch_len), int(cfg.patch_len), int(cfg.window_size / cfg.patch_len), head_type='classification')
y = model(next(iter(fog_train_loader))[0].float())
print("HERE", y.shape)

# get the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# ## Fine - Tuning
# In[191]:


class FOGModule(pl.LightningModule):

    def __init__(self, model, optimizer_name):
        """
        Inputs:
            model_name - Name of the model to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        #self.save_hyperparameters(ignore=["model"])
        self.save_hyperparameters()
        self.lr = cfg.lr
        # Create model
        self.model = model
        # Create loss module
        self.loss_module = nn.BCEWithLogitsLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((int(cfg.batch_size), int(cfg.window_size/cfg.patch_len), 3, int(cfg.patch_len)), dtype=torch.float32)
        self.val_true = None
        self.val_pred = None

    def forward(self, past):
        # Forward function that is run when visualizing the graph
        return self.model(past)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=cfg.gamma)
        else:
            assert False, f"Unknown optimizer: \"{self.hparams.optimizer_name}\""

        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode="min")
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {
          "optimizer": optimizer,
          "lr_scheduler": {
              "scheduler": scheduler,
              "monitor": "val_loss",
              "interval": "epoch",
              "frequency": 1
          }
        }
        #return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        x, y, t = batch
        x = x.float()
        y = y.float()
        preds = self.model(x)
        loss = self.loss_module(preds, y)
        acc = (preds.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('train_acc', acc, sync_dist=True)
        
        with torch.no_grad():
            ap = self.average_precision_score(y, preds)
        self.log('train_ap0', ap[0], sync_dist=True)
        self.log('train_ap1', ap[1], sync_dist=True)
        self.log('train_ap2', ap[2], sync_dist=True)
        self.log('train_ap', sum(ap)/3, sync_dist=True)

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        #self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss  # Return tensor to call ".backward" on

    def on_train_epoch_end(self):
        avg_precision = self.trainer.logged_metrics['train_ap0'].nanmean()
        self.log('train0_precision', avg_precision, sync_dist=True)
        avg_precision = self.trainer.logged_metrics['train_ap1'].nanmean()
        self.log('train1_precision', avg_precision, sync_dist=True)
        avg_precision = self.trainer.logged_metrics['train_ap2'].nanmean()
        self.log('train2_precision', avg_precision, sync_dist=True)
        avg_precision = self.trainer.logged_metrics['train_ap'].nanmean()
        self.log('avg_train_precision', avg_precision, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y, t = batch
        x = x.float()
        y = y.float()
        preds = self.model(x)
 
        if self.val_true is None:
            self.val_true = y
            self.val_pred = preds
        else:
            self.val_true = torch.cat((self.val_true, y), dim=0)
            self.val_pred = torch.cat((self.val_pred, preds), dim=0)

        loss = self.loss_module(preds, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return {'loss': loss, 'y_pred': preds, 'y': y}
    
    def on_validation_epoch_end(self):
        
        # avg_precision = self.trainer.logged_metrics['val_ap0'].nanmean()
        # self.log('val0_precision', avg_precision)
        # avg_precision = self.trainer.logged_metrics['val_ap1'].nanmean()
        # self.log('val1_precision', avg_precision)
        # avg_precision = self.trainer.logged_metrics['val_ap2'].nanmean()
        # self.log('val2_precision', avg_precision)
        # avg_precision = self.trainer.logged_metrics['val_ap'].nanmean()
        # self.log('avg_val_precision', avg_precision)

        acc = (self.val_true.argmax(dim=-1) == self.val_pred.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc, sync_dist=True)
        avg_precision = self.average_precision_score(self.val_true, self.val_pred)
        self.log('val0_precision', avg_precision[0], sync_dist=True)
        self.log('val1_precision', avg_precision[1], sync_dist=True)
        self.log('val2_precision', avg_precision[2], sync_dist=True)
        self.log('avg_val_precision', sum(avg_precision)/3, sync_dist=True)
        self.val_true = None
        self.val_pred = None

    def test_step(self, batch, batch_idx):
        x, y, t = batch
        x = x.float()
        y = y.float()
        preds = self.model(x)
 
        loss = self.loss_module(preds, y)
        if self.val_true is None:
            self.val_true = y
            self.val_pred = preds
        else:
            self.val_true = torch.cat((self.val_true, y), dim=0)
            self.val_pred = torch.cat((self.val_pred, preds), dim=0)
        return {'loss': loss, 'y_pred': preds, 'y': y}
    
    def on_test_epoch_end(self) -> None:
        acc = (self.val_true.argmax(dim=-1) == self.val_pred.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc, sync_dist=True)
        avg_precision = self.average_precision_score(self.val_true, self.val_pred)
        self.log('val0_precision', avg_precision[0], sync_dist=True)
        self.log('val1_precision', avg_precision[1], sync_dist=True)
        self.log('val2_precision', avg_precision[2], sync_dist=True)
        self.log('avg_val_precision', sum(avg_precision)/3, sync_dist=True)
        self.val_true = None
        self.val_pred = None
    
    def average_precision_score(self, y_true, y_pred):
        target = y_true.argmax(dim=-1)
        return multiclass_average_precision(y_pred, target, num_classes=3, average=None)
        


# In[192]:

class ConfusionMatrixCallback(pl.Callback):
    def __init__(self, num_classes, task="multiclass"):
        super().__init__()
        self.conf_matrix = torchmetrics.ConfusionMatrix(num_classes=num_classes, task=task).to(cfg.device)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        # Get the predicted labels and ground truth labels from the batch
        y_pred, y_true = outputs['y_pred'], outputs['y']

        # Update the confusion matrix with the current batch
        self.conf_matrix.update(y_pred, y_true)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Compute the confusion matrix for the entire validation set
        matrix = self.conf_matrix.compute().detach().cpu()

        # Print the confusion matrix
        print('Confusion matrix:')
        print(matrix)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        # Get the predicted labels and ground truth labels from the batch
        y_pred, y_true = outputs['y_pred'], outputs['y']

        # Update the confusion matrix with the current batch
        self.conf_matrix.update(y_pred, y_true)

    def on_test_end(self, trainer, pl_module):
        # Compute the confusion matrix for the entire test set
        matrix = self.conf_matrix.compute().detach().cpu()

        # Print the confusion matrix
        print('Confusion matrix:')
        print(matrix)

def train_model(module, model, train_loader, val_loader, test_loader, save_name = None, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    # Create a PyTorch Lightning trainer with the generation callback
    conf_matrix_callback = ConfusionMatrixCallback(num_classes=3, task="multiclass")
    trainer = pl.Trainer(default_root_dir=os.path.join(cfg.CHECKPOINT_PATH, save_name),                          # Where to save models
                         accelerator="gpu" if str(cfg.device).startswith("cuda") else "cpu",                     # We run on a GPU (if possible)
                         devices=torch.cuda.device_count() if str(cfg.device).startswith("cuda") else 1,         # How many GPUs/CPUs we want to use (1 is enough for the notebooks)
                         max_epochs=cfg.num_epochs,                                                                     # How many epochs to train for if no patience is set
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="avg_val_precision"),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                    LearningRateMonitor("epoch")],                                           # Log learning rate every epoch
                         enable_progress_bar=True,                                                          # Set to False if you do not want a progress bar
                         logger = True,
                         strategy=DDPStrategy(find_unused_parameters=True),
                         val_check_interval=0.5,
                         log_every_n_steps=50)                                                           
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = True

    # log hyperparameters, including model and custom parameters
    del cfg.hparams["milestones"] 
    trainer.logger.log_metrics(cfg.hparams)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(cfg.CHECKPOINT_PATH, save_name + f"/{cfg.model_hidden}_{cfg.model_nblocks}_final.pt")
    print(f"Pretrained file {pretrained_filename}")
    if cfg.hparams["use_pretrained"] and os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model.load_state_dict(torch.load(pretrained_filename)) # Automatically loads the model with the saved hyperparameters
    lmodel = module(model, **kwargs)

    # tune learning rate
    # print("Tuning learning rate...")
    # tuner = Tuner(trainer)
    # # Run learning rate finder
    # lr_finder = tuner.lr_find(lmodel, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    # print(f"New learning rate: {new_lr}")
    # print("Tuning done.")
    
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


model, trainer, result = train_model(FOGModule, model, fog_train_loader, fog_valid_loader, fog_valid_loader, save_name="FOGPatchTST", optimizer_name="Adam")
print(json.dumps(cfg.hparams, sort_keys=True, indent=4))
print(json.dumps(result, sort_keys=True, indent=4))


# ## Submission

# In[ ]:


model = FOGModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
model.to(cfg.device)
model.eval()

test_defog_paths = glob.glob(f"{cfg.DATA_DIR}test/defog/*.csv")
test_tdcsfog_paths = glob.glob(f"{cfg.DATA_DIR}test/tdcsfog/*.csv")
test_fpaths = [(f, 'de') for f in test_defog_paths] + [(f, 'tdcs') for f in test_tdcsfog_paths]

test_dataset = FOGPatchDataSet(test_fpaths, cfg, split="test")
test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size) #, num_workers=cfg.num_workers)

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
