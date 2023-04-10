#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time


# In[2]:


from icecream import ic
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd


# In[3]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


# In[4]:


from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, average_precision_score


# In[5]:


class Config:
    DATA_DIR = '../tlvmc-parkinsons-freezing-gait-prediction/'
    TRAIN_DIR = '../tlvmc-parkinsons-freezing-gait-prediction/train/'
    TDCSFOG_DIR = '../tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog/'
    DEFOG_DIR = '../tlvmc-parkinsons-freezing-gait-prediction/train/defog/'
    CHECKPOINT_PATH = './checkpoints/'

    batch_size = 2048
    window_size = 256
    window_future = 32
    window_past = window_size - window_future

    model_dropout = 0.3
    model_hidden = 768
    model_nblocks = 2

    lr = 0.001
    num_epochs = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    feature_list = ['AccV', 'AccML', 'AccAP']
    label_list = ['StartHesitation', 'Turn', 'Walking']

cfg = Config()


# In[6]:


print(cfg.device)


# ## Data - Preprocessing

# In[7]:


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


# In[8]:


# Analysis of positive instances in each fold of our CV folds

SH = []
T = []
W = []

# Here I am using the metadata file available during training. Since the code will run again during submission, if 
# I used the usual file from the competition folder, it would have been updated with the test files too.
metadata = pd.read_csv("../tlvmc-parkinsons-freezing-gait-prediction/tdcsfog_metadata.csv")

for f in tqdm(metadata['Id']):
    fpath = f"../tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog/{f}.csv"
    df = pd.read_csv(fpath)
    
    SH.append(np.sum(df['StartHesitation']))
    T.append(np.sum(df['Turn']))
    W.append(np.sum(df['Walking']))
    
print(f"32 files have positive values in all 3 classes")

metadata['SH'] = SH
metadata['T'] = T
metadata['W'] = W

sgkf = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
for i, (train_index, valid_index) in enumerate(sgkf.split(X=metadata['Id'], y=[1]*len(metadata), groups=metadata['Subject'])):
    print(f"Fold = {i}")
    train_ids = metadata.loc[train_index, 'Id']
    valid_ids = metadata.loc[valid_index, 'Id']
    
    print(f"Length of Train = {len(train_ids)}, Length of Valid = {len(valid_ids)}")
    n1_sum = metadata.loc[train_index, 'SH'].sum()
    n2_sum = metadata.loc[train_index, 'T'].sum()
    n3_sum = metadata.loc[train_index, 'W'].sum()
    print(f"Train classes: {n1_sum:,}, {n2_sum:,}, {n3_sum:,}")
    
    n1_sum = metadata.loc[valid_index, 'SH'].sum()
    n2_sum = metadata.loc[valid_index, 'T'].sum()
    n3_sum = metadata.loc[valid_index, 'W'].sum()
    print(f"Valid classes: {n1_sum:,}, {n2_sum:,}, {n3_sum:,}")
    
# # FOLD 2 is the most well balanced


# In[9]:


# The actual train-test split (based on Fold 2)

metadata = pd.read_csv(cfg.DATA_DIR + "tdcsfog_metadata.csv")
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
        
train_fpaths = [f"{cfg.DATA_DIR}train/tdcsfog/{_id}.csv" for _id in train_ids]
valid_fpaths = [f"{cfg.DATA_DIR}train/tdcsfog/{_id}.csv" for _id in valid_ids]


# In[10]:


#import glob module 
import glob
# get all the file paths in to list for the test directory
test_fpaths = glob.glob(f"{cfg.DATA_DIR}test/tdcsfog/*.csv")
test_fpaths


# # TODO: Generalize preprocessing

# In[11]:


tdcsfog_train = FOGDataset(train_fpaths)
tdcsfog_train_loader = DataLoader(tdcsfog_train, batch_size=cfg.batch_size, shuffle=True)


# In[12]:


tdcsfog_valid = FOGDataset(valid_fpaths)
tdcsfog_valid_loader = DataLoader(tdcsfog_valid, batch_size=cfg.batch_size)


# In[13]:


tdcsfog_test = FOGDataset(test_fpaths)
tdcsfog_test_loader = DataLoader(tdcsfog_test, batch_size=cfg.batch_size)


# ## Model

# In[14]:


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

class FOGTransformer(nn.Module):
    def __init__(self, p=cfg.model_dropout, dim=cfg.model_hidden, nblocks=cfg.model_nblocks):
        super(FOGTransformer, self).__init__()
        self.dropout = nn.Dropout(p)
        self.in_layer = nn.Linear(cfg.window_size*3, dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1, mask_check=False)

        self.out_layer = nn.Linear(dim, 3)

    def forward(self, x):
        x = x.view(-1, cfg.window_size*3)
        x = self.in_layer(x)
        x = self.transformer(x)
        x = self.out_layer(x)
        return x


# In[15]:


# get the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = FOGTransformer()
print(f'The model has {count_parameters(model):,} trainable parameters')
model = FOGModel()
print(f'The model has {count_parameters(model):,} trainable parameters')


# In[60]:

"""
tdcsfog_train_loader = DataLoader(tdcsfog_train, batch_size=cfg.batch_size, shuffle=True)

model = FOGModel()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
criterion = nn.BCEWithLogitsLoss()
soft = nn.Softmax(dim=-1)

def train(model, optimizer, criterion, train_loader):
    for x, y in tqdm(train_loader):
        ic(y)
        ic(x.shape, y.shape)
        #ic(x, y)
        # single forward pass
        # cast x to the correct data type
        x = x.float()
        y_hat = model(x)
        ic(y_hat)
        ic(soft(y_hat))
        ic(y_hat.shape)
        ic(y_hat.argmax(dim=-1))
        # calculate loss
        loss = criterion(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        # calculate gradients
        loss.backward()
        # update weights
        optimizer.step()
        # print out the loss using ic
        ic(loss.item())
        break

train(model, optimizer, criterion, tdcsfog_train_loader)
"""


# In[68]:


model_dict = {}

def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, f"Unknown model name \"{model_name}\". Available models are: {str(model_dict.keys())}"

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
        self.example_input_array = torch.zeros((1, 256, 3), dtype=torch.float32)

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
            optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        past, future = batch
        past = past.float()
        future = future.float()
        preds = self.model(past)
        loss = self.loss_module(preds, future)
        acc = (preds.argmax(dim=-1) == future.argmax(dim=-1)).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        past, future = batch
        past = past.float()
        future = future.float()
        preds = self.model(past)
        acc = (future.argmax(dim=-1) == preds.argmax(dim=-1)).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        past, future = batch
        past = past.float()
        future = future.float()
        preds = self.model(past)
        acc = (future == preds.argmax(dim=-1)).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log('test_acc', acc)


# In[65]:


def train_model(module, model, train_loader, val_loader, test_loader, save_name = None, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(cfg.CHECKPOINT_PATH, save_name),                          # Where to save models
                         accelerator="gpu", # if str(cfg.device).startswith("cuda") else "cpu",                     # We run on a GPU (if possible)
                         devices=1,                                                                          # How many GPUs/CPUs we want to use (1 is enough for the notebooks)
                         max_epochs=180,                                                                     # How many epochs to train for if no patience is set
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                    LearningRateMonitor("epoch")],                                           # Log learning rate every epoch
                         enable_progress_bar=True,
                         logger = True)                                                           # Set to False if you do not want a progress bar
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = True # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(cfg.CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = module.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(42) # To be reproducable
        lmodel = module(model, **kwargs)
        trainer.fit(lmodel, train_loader, val_loader)
        lmodel = module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    #test_result = trainer.test(model, test_loader, verbose=False)
    #result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return lmodel, None


# In[69]:


model = FOGModel()
model, result = train_model(FOGModule, model, tdcsfog_train_loader, tdcsfog_valid_loader, tdcsfog_test_loader, save_name="FOGModel", optimizer_name="Adam", optimizer_hparams={"lr": 0.001, "weight_decay": 0.0001})


# ## Pre - Training

# In[ ]:





# In[ ]:





# ## Fine - Tuning

# 
