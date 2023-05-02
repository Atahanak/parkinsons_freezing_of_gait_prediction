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
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchmetrics.functional.classification import multiclass_average_precision

torch.set_float32_matmul_precision('high')

# In[164]:


from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import average_precision_score


# In[165]:


class Config:
    KAGGLE = False
    ROOT_READ = '../'
    ROOT_WRITE = '../'
    if KAGGLE:
        ROOT_READ = '/kaggle/input/'
        ROOT_WRITE = '/kaggle/working/'
    DATA_DIR = f'{ROOT_READ}tlvmc-parkinsons-freezing-gait-prediction/'
    TRAIN_DIR = f'{ROOT_READ}tlvmc-parkinsons-freezing-gait-prediction/train/'
    TDCSFOG_DIR = f'{ROOT_READ}tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog/'
    DEFOG_DIR = f'{ROOT_READ}tlvmc-parkinsons-freezing-gait-prediction/train/defog/'
    CHECKPOINT_PATH = f'{ROOT_WRITE}checkpoints/'
    PARAMS_PATH = f'./config.json'
    if KAGGLE:
        PARAMS_PATH = '/' # TODO


    with open(PARAMS_PATH) as f:
        hparams = json.load(f)

    batch_size = hparams["batch_size"]
    window_size = hparams["window_size"]
    window_future = hparams["window_future"]
    window_past = window_size - window_future
    wx = hparams["wx"]

    model_dropout = hparams["model_dropout"]
    model_hidden = hparams["model_hidden"]
    model_nblocks = hparams["model_nblocks"]
    model_nhead = hparams["model_nhead"]

    lr = hparams["lr"]
    milestones = hparams["milestones"]
    gamma = hparams["gamma"]

    num_epochs = hparams["num_epochs"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 32 if torch.cuda.is_available() else 1

    feature_list = ['AccV', 'AccML', 'AccAP']
    label_list = ['StartHesitation', 'Turn', 'Walking']

cfg = Config()
cfg.num_workers


# In[166]:


cfg.device


# ## Data - Preprocessing

# In[167]:


"""
class FOGDataset(Dataset):
    def __init__(self, fpaths, scale=9.806, test=False):
        super(FOGDataset, self).__init__()
        tm = time.time()
        self.test = test
        self.fpaths = fpaths
        self.f_ids = [os.path.basename(f)[:-4] for f in self.fpaths]
        self.curr_df_idx = 0
        self.curr_row_idx = 0
        self.dfs = [np.array(pd.read_csv(f)) for f in fpaths]
        self.end_indices = []
        self.scale = scale
        
        self.length = 0
        for df in self.dfs:
            self.length += df.shape[0]
            self.end_indices.append(self.length)
            
        print(f"Dataset initialized in {time.time() - tm} secs!")
        
    def pad(self, df, time_start):
        if df.shape[0] == cfg.window_size:
            return df
        
        npad = cfg.window_size - df.shape[0]
        padzeros = np.zeros((npad, 3))
        if time_start <= 0:
            df = np.concatenate((padzeros, df), axis=0)
        else:
            df = np.concatenate((df, padzeros), axis=0)
        return df
            
    def __getitem__(self, index):
        for i,e in enumerate(self.end_indices):
            if index >= e:
                continue
            df_idx = i
            break
            
        curr_df = self.dfs[i]
        row_idx = curr_df.shape[0] - (self.end_indices[i] - index)
        _id = self.f_ids[i] + "_" + str(row_idx)
        
        x = self.pad(curr_df[row_idx-cfg.window_past:row_idx+cfg.window_future, 1:4], row_idx-cfg.window_past )
        x = torch.tensor(x)/self.scale
        
        if self.test == True:
            return _id, x
        
        y = curr_df[row_idx, -3:].astype('float')
        y = torch.tensor(y)
        
        return x, y
    
    def __len__(self):
        return self.length
"""
class FOGDataset(Dataset):
    def __init__(self, fpaths, scale=9.806, split="train"):
        super(FOGDataset, self).__init__()
        tm = time.time()
        self.split = split
        self.scale = scale
        
        self.fpaths = fpaths
        self.dfs = [self.read(f[0], f[1]) for f in fpaths]
        self.f_ids = [os.path.basename(f[0])[:-4] for f in self.fpaths]
        
        self.end_indices = []
        self.shapes = []
        _length = 0
        for df in self.dfs:
            self.shapes.append(df.shape[0])
            _length += df.shape[0]
            self.end_indices.append(_length)
        
        self.dfs = np.concatenate(self.dfs, axis=0).astype(np.float16)
        self.length = self.dfs.shape[0]
        
        shape1 = self.dfs.shape[1]
        
        self.dfs = np.concatenate([np.zeros((cfg.wx*cfg.window_past, shape1)), self.dfs, np.zeros((cfg.wx*cfg.window_future, shape1))], axis=0)
        print(f"Dataset initialized in {time.time() - tm} secs!")
        gc.collect()
        
    def read(self, f, _type):
        df = pd.read_csv(f)
        if self.split == "test":
            return np.array(df)
        
        if _type =="tdcs":
            df['Valid'] = 1
            df['Task'] = 1
            df['tdcs'] = 1
        else:
            df['tdcs'] = 0
        
        return np.array(df)
            
    def __getitem__(self, index):
        if self.split == "train":
            row_idx = random.randint(0, self.length-1) + cfg.wx*cfg.window_past
        elif self.split == "test":
            for i,e in enumerate(self.end_indices):
                if index >= e:
                    continue
                df_idx = i
                break

            row_idx_true = self.shapes[df_idx] - (self.end_indices[df_idx] - index)
            _id = self.f_ids[df_idx] + "_" + str(row_idx_true)
            row_idx = index + cfg.wx*cfg.window_past
        else:
            row_idx = index + cfg.wx*cfg.window_past
            
        #scale = 9.806 if self.dfs[row_idx, -1] == 1 else 1.0
        x = self.dfs[row_idx - cfg.wx*cfg.window_past : row_idx + cfg.wx*cfg.window_future, 1:4]
        x = x[::cfg.wx, :][::-1, :]
        x = torch.tensor(x.astype('float'))#/scale
        
        t = self.dfs[row_idx, -3]*self.dfs[row_idx, -2]
        
        if self.split == "test":
            return _id, x, t
        
        y = self.dfs[row_idx, 4:7].astype('float')
        y = torch.tensor(y)
        
        return x, y, t
    
    def __len__(self):
        return self.length


# In[168]:


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


fog_train = FOGDataset(train_fpaths)
fog_train_loader = DataLoader(fog_train, batch_size=cfg.batch_size, shuffle=True, num_workers=16)


# In[172]:


fog_valid = FOGDataset(valid_fpaths)
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


# ## Model

# In[1]:


def _block(in_features, out_features, drop_rate):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(drop_rate)
    )

class FOGModel(nn.Module):
    def __init__(self, p=cfg.model_dropout, dim=cfg.model_hidden, nblocks=cfg.model_nblocks):
        super(FOGModel, self).__init__()
        self.hparams = {}
        self.dropout = nn.Dropout(p)
        self.in_layer = nn.Linear(cfg.window_size*3, dim)
        self.blocks = nn.Sequential(*[_block(dim, dim, p) for _ in range(nblocks)])
        self.out_layer = nn.Linear(dim, 3)
        
    def forward(self, x):
        x = x.view(-1, cfg.window_size*3)
        x = self.in_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_layer(x)
        return x

class FOGTransformerEncoder(nn.Module):
    def __init__(self, state="finetune", p=cfg.model_dropout, dim=cfg.model_hidden, nblocks=cfg.model_nblocks):
        super(FOGTransformerEncoder, self).__init__()
        self.hparams = {}
        self.state = state
        self.dropout = nn.Dropout(p)
        self.in_layer = nn.Linear(cfg.window_size*3, dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=cfg.model_nhead, dim_feedforward=dim)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=nblocks, mask_check=False)

        self.out_layer_pretrain = nn.Linear(dim, cfg.window_future * 3)
        self.out_layer_finetune = nn.Linear(dim, 3)

    def forward(self, x):
        x = x.view(-1, cfg.window_size*3)
        x = self.in_layer(x)
        x = self.transformer(x)
        if self.state == "pre-train":
            x = self.out_layer_pretrain(x)
        else:
            x = self.out_layer_finetune(x)
        return x


# In[190]:


# get the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(FOGTransformerEncoder()):,} trainable parameters')
print(f'The model has {count_parameters(FOGModel()):,} trainable parameters')


# In[177]:


"""
tdcsfog_train_loader = DataLoader(tdcsfog_train, batch_size=cfg.batch_size, shuffle=True)

model = FOGModel()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
criterion = nn.BCEWithLogitsLoss()
soft = nn.Softmax(dim=-1)

# def average_precision_score(y_true, y_pred):
#         # average precision with pytorch
#         y = y_true.argmax(dim=-1)
#         average_precision = AveragePrecision(task="multiclass", num_classes=3, average=None)
#         return average_precision(y_pred, y)

def train(model, optimizer, criterion, train_loader):
    for x, y in tqdm(train_loader):
        # print(y)
        # print(x.shape, y.shape)
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
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        # calculate gradients
        loss.backward()
        # update weights
        optimizer.step()
        # print out the loss using ic
        #print(loss.item())
        #print(acc.item())
        print(y)

        with torch.no_grad():
            print(average_precision_score(y, y_hat, average=None))
            print(multiclass_average_precision(y_hat, y.argmax(-1), num_classes=3, average=None))
        break

def validation(model, criterion, valid_loader):
    lol = []
    lil = []
    c = 0
    for x, y in tqdm(valid_loader):
        # single forward pass
        # cast x to the correct data type
        x = x.float()
        # disable gradient calculation
        with torch.no_grad():
            y_hat = model(x)
        print(y_hat)
        #print(y_hat.argmax(dim=-1))
        print(y)
        lil = lil + y_hat.tolist()
        lol = lol + y.tolist()
        #print(y.argmax(dim=-1))

        # calculate loss
        loss = criterion(y_hat, y)
        acc = (lil.argmax(dim=-1) == lol.argmax(dim=-1)).float().mean()
        # print out the loss using ic
        #print(loss.item())
        #print(acc.item())
        print(average_precision_score(y, y_hat, average=None))
        print(multiclass_average_precision(y_hat, y.argmax(-1), num_classes=3, average=None))
        c += 1
        if c == 3:
            break
    print(lil)
    print(lol)
"""


# ## Fine - Tuning

# In[191]:


class FOGModule(pl.LightningModule):

    def __init__(self, model, optimizer_name, optimizer_hparams):
        """
        Inputs:
            model_name - Name of the model to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = model
        # Create loss module
        self.loss_module = nn.BCEWithLogitsLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, cfg.window_size, 3), dtype=torch.float32)
        self.val_true = None
        self.val_pred = None

    def forward(self, past):
        # Forward function that is run when visualizing the graph
        return self.model(past)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = torch.optim.AdamW(
                self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f"Unknown optimizer: \"{self.hparams.optimizer_name}\""

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.milestones, gamma=cfg.gamma)
        return [optimizer], [scheduler]

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
        # By default logs it per epoch (weighted average over batches)
        # with torch.no_grad():
        #     ap = self.average_precision_score(future, preds)
        # self.log('val_ap0', ap[0])
        # self.log('val_ap1', ap[1])
        # self.log('val_ap2', ap[2])
        # self.log('val_ap', sum(ap)/3)
        # self.log('val_ap33', sum(ap)/3, on_step=True)
    
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
 
        if self.val_true is None:
            self.val_true = y
            self.val_pred = preds
        else:
            self.val_true = torch.cat((self.val_true, y), dim=0)
            self.val_pred = torch.cat((self.val_pred, preds), dim=0)
    
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


def train_model(module, model, train_loader, val_loader, test_loader, save_name = None, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(cfg.CHECKPOINT_PATH, save_name),                          # Where to save models
                         accelerator="gpu" if str(cfg.device).startswith("cuda") else "cpu",                     # We run on a GPU (if possible)
                         devices=torch.cuda.device_count() if str(cfg.device).startswith("cuda") else 1,         # How many GPUs/CPUs we want to use (1 is enough for the notebooks)
                         max_epochs=cfg.num_epochs,                                                                     # How many epochs to train for if no patience is set
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="avg_val_precision"),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                    LearningRateMonitor("epoch")],                                           # Log learning rate every epoch
                         enable_progress_bar=True,                                                          # Set to False if you do not want a progress bar
                         logger = True,
                         strategy=DDPStrategy(find_unused_parameters=True),
                         # val_check_interval=0.5,
                         log_every_n_steps=50)                                                           
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = True

    # log hyperparameters, including model and custom parameters
    model.hparams.update(cfg.hparams)
    trainer.logger.log_hyperparams(model.hparams)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(cfg.CHECKPOINT_PATH, save_name + f"/{cfg.model_hidden}_{cfg.model_nblocks}_final.pt")
    print(f"Pretrained file {pretrained_filename}")
    if cfg.hparams["use_pretrained"] and os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model.load_state_dict(torch.load(pretrained_filename)) # Automatically loads the model with the saved hyperparameters
    lmodel = module(model, **kwargs)
    
    pl.seed_everything(42) # To be reproducable
    trainer.fit(lmodel, train_loader, val_loader)
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


model = FOGTransformerEncoder()
model, trainer, result = train_model(FOGModule, model, fog_train_loader, fog_valid_loader, fog_valid_loader, save_name="FOGTransformerEncoder", optimizer_name="Adam", optimizer_hparams={"lr": cfg.lr, "weight_decay": cfg.gamma})
print(json.dumps(cfg.hparams, sort_keys=True, indent=4))
result


# ## Submission

# In[ ]:


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

