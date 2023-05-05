#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import time
import random
import gc
import glob
import json


# In[7]:


from tqdm.notebook import tqdm
import numpy as np
import pandas as pd


# In[8]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from torchmetrics.functional.classification import multiclass_average_precision
print(pl.__version__)

torch.set_float32_matmul_precision('high')

# In[9]:


class Config:
    KAGGLE = False
    ROOT_READ = '../'
    ROOT_WRITE = '../'
    if KAGGLE:
        ROOT_READ = '/kaggle/input/'
        ROOT_WRITE = '/kaggle/working/'
    DATA_DIR = f'{ROOT_READ}tlvmc-parkinsons-freezing-gait-prediction/'
    TRAIN_DIR = f'{ROOT_READ}tlvmc-parkinsons-freezing-gait-prediction/train/'
    CHECKPOINT_PATH = f'{ROOT_WRITE}checkpoints/'
    PARAMS_PATH = f'./pretrain-config.json'
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
    num_workers = os.cpu_count()

    feature_list = ['AccV', 'AccML', 'AccAP']
    label_list = ['StartHesitation', 'Turn', 'Walking']

cfg = Config()
print(vars(Config))


# ## Data - Preprocessing

# In[10]:


class FOGDataset(Dataset):
    def __init__(self, fpaths, scale=9.806, split="train", state="fine-tune"):
        super(FOGDataset, self).__init__()
        tm = time.time()
        self.split = split
        self.state = state
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
            print(df.shape[0], _length)
        
        self.dfs = np.concatenate(self.dfs, axis=0).astype(np.float16)
        self.length = self.dfs.shape[0]
        
        shape1 = self.dfs.shape[1]
        
        self.dfs = np.concatenate([np.zeros((cfg.wx*cfg.window_past, shape1)), self.dfs, np.zeros((2*cfg.wx*cfg.window_future, shape1))], axis=0)
        print(f"Dataset initialized in {time.time() - tm} secs!")
        gc.collect()
        
    def read(self, f, _type):
        print(f"Reading file {f}...")
        if self.state == "pre-train":
            df = pd.read_parquet(f)
        elif self.state == "fine-tune": 
            df = pd.read_csv(f)
            
        if self.split == "test" or self.state == "pre-train":
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
        x = torch.tensor(x.astype('float'))
        
        t = self.dfs[row_idx, -3]*self.dfs[row_idx, -2]
        
        y = self.dfs[row_idx + cfg.wx*cfg.window_future : row_idx + 2*cfg.wx*cfg.window_future, 1:4]
        y = y[::cfg.wx, :][::-1, :]
        y = torch.tensor(y.astype('float'))
        return x, y, t

    def __len__(self):
        return self.length

# ## Model

# In[11]:


def _block(in_features, out_features, drop_rate):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(drop_rate)
    )

class FOGModel(nn.Module):
    def __init__(self, state="finetune", p=cfg.model_dropout, dim=cfg.model_hidden, nblocks=cfg.model_nblocks):
        super(FOGModel, self).__init__()
        self.hparams = {}
        self.state = state
        self.dropout = nn.Dropout(p)
        self.in_layer = nn.Linear(cfg.window_size*3, dim)
        self.blocks = nn.Sequential(*[_block(dim, dim, p) for _ in range(nblocks)])

        self.out_layer_pretrain = nn.Linear(dim, cfg.window_future * 3)
        self.out_layer_finetune = nn.Linear(dim, 3)
        
    def forward(self, x):
        x = x.view(-1, cfg.window_size*3)
        x = self.in_layer(x)
        for block in self.blocks:
            x = block(x)
        if self.state == "pretrain":
            x = self.out_layer_pretrain(x)
        else:
            x = self.out_layer_finetune(x)
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
        if self.state == "pretrain":
            x = self.out_layer_pretrain(x)
        else:
            x = self.out_layer_finetune(x)
        return x


# In[ ]:


# get the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'FOGTransformerEncoder has {count_parameters(FOGTransformerEncoder()):,} trainable parameters')
print(f'FOGModel has {count_parameters(FOGModel()):,} trainable parameters')


# # Pre-Training

# In[12]:

print("Initializing dataset...")
pretrain_paths = [(f, 'unlabeled') for f in glob.glob(f"{cfg.DATA_DIR}unlabeled/*.parquet")]
fog_pretrain = FOGDataset(pretrain_paths, state="pre-train")
fog_train_loader = DataLoader(fog_pretrain, batch_size=cfg.batch_size, num_workers=16) #cfg.num_workers)
print("Dataset size:", fog_pretrain.__len__())
print("Number of batches:", len(fog_train_loader))
print("Batch size:", fog_train_loader.batch_size)
print("Total size:", len(fog_train_loader) * fog_train_loader.batch_size)

item = next(iter(fog_train_loader))

# In[13]:


gc.collect()


# In[14]:


class PreTrainingFogModule(pl.LightningModule):
    def __init__(self, model, optimizer_name, optimizer_hparams):
        super(PreTrainingFogModule, self).__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, cfg.window_size, 3), dtype=torch.float32)

        self.model = model
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        x = x.float()
        y = y.float()
        y_hat = self.model(x)
        y = y.view(y.shape[0], -1)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), **self.hparams.optimizer_hparams)
        return optimizer


# In[15]:


def pretrain_model(module, model, train_loader, save_name = None, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(cfg.CHECKPOINT_PATH, save_name),                          # Where to save models
                         accelerator="gpu" if str(cfg.device).startswith("cuda") else "cpu",                     # We run on a GPU (if possible)
                         devices=torch.cuda.device_count() if str(cfg.device).startswith("cuda") else 1,                                                                          # How many GPUs/CPUs we want to use (1 is enough for the notebooks)
                         max_epochs=cfg.num_epochs,                                                                     # How many epochs to train for if no patience is set
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
    model.hparams.update(cfg.hparams)
    del model.hparams["milestones"] # = str(model.hparams["milestones"])
    trainer.logger.log_metrics(model.hparams)

    # Check whether pretrained model exists. If yes, load it and skip training
    pl.seed_everything(42) # To be reproducable
    lmodel = module(model, **kwargs)
    trainer.fit(lmodel, train_loader)
    lmodel = module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training
    
    #save model ready to be loaded for finetuning
    pretrained_file_name = os.path.join(cfg.CHECKPOINT_PATH, save_name, f"{cfg.model_hidden}_{cfg.model_nblocks}_final.pt")
    print(f"Saving pretrained model to: {pretrained_file_name}")
    torch.save(lmodel.model.state_dict(), pretrained_file_name)

    train_loss = trainer.logged_metrics["train_loss"]
    result = {
        "train_loss": train_loss.item()
    }

    return lmodel, trainer, result


# In[16]:


print(f"# of devices: {torch.cuda.device_count()}")
model = FOGModel(state="pretrain")
model, trainer, result = pretrain_model(PreTrainingFogModule, model, fog_train_loader, save_name="FOGModel", optimizer_name="Adam", optimizer_hparams={"lr": cfg.lr, "weight_decay": cfg.gamma})
print(json.dumps(cfg.hparams, sort_keys=True, indent=4))
print(json.dumps(result, sort_keys=True, indent=4))


# In[ ]:


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

