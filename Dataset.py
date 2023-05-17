__all__ = ['FOGPatchDataSet', 'FOGFinetuneDataset', 'FOGPretrainDataset']

import os
import gc
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

import torch
from torch.utils.data import Dataset

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
        if df.shape[0] == self.cfg.window_size:
            return df
        
        npad = self.cfg.window_size - df.shape[0]
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
        
        x = self.pad(curr_df[row_idx-self.cfg.window_past:row_idx+self.cfg.window_future, 1:4], row_idx-self.cfg.window_past )
        x = torch.tensor(x)/self.scale
        
        if self.test == True:
            return _id, x
        
        y = curr_df[row_idx, -3:].astype('float')
        y = torch.tensor(y)
        
        return x, y
    
    def __len__(self):
        return self.length
"""

class FOGPatchDataSet(Dataset):
  def __init__(self, fpaths, cfg, scale=9, split='train', task='pretrain'):
    super(FOGPatchDataSet).__init__()
    tm = time.time()
    self.cfg = cfg
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
    print(f"Reading {f}...")
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
        row_idx = random.randint(0, self.length-1) + self.cfg.wx*self.cfg.window_past
    elif self.split == "test":
        for i,e in enumerate(self.end_indices):
            if index >= e:
                continue
            df_idx = i
            break

        row_idx_true = self.shapes[df_idx] - (self.end_indices[df_idx] - index)
        _id = self.f_ids[df_idx] + "_" + str(row_idx_true)
        row_idx = index + self.cfg.wx*self.cfg.window_past
    else:
        row_idx = index + self.cfg.wx*self.cfg.window_past
        
    #scale = 9.806 if self.dfs[row_idx, -1] == 1 else 1.0
    x = self.dfs[row_idx - self.cfg.wx*self.cfg.window_past : row_idx + self.cfg.wx*self.cfg.window_future, 1:4]
    x = x[::self.cfg.wx, :][::-1, :]
    #print('DEBUG', x.shape, x.shape[0])
    num_patch = x.shape[0] // (self.cfg.patch_len)
    x = x.reshape(num_patch, 3, self.cfg.patch_len)
    x = torch.tensor(x.astype('float'))#/scale
    #print('DEBUG', x.shape, x.shape[0])

    t = self.dfs[row_idx, -3]*self.dfs[row_idx, -2]
    
    if self.split == "test":
        return _id, x, t
    
    y = self.dfs[row_idx, 4:7].astype('float')
    y = torch.tensor(y)
    
    return x, y, t

  def __len__(self):
    return self.length  

class FOGFinetuneDataset(Dataset):
    def __init__(self, fpaths, cfg, scale=9.806, split="train"):
        super(FOGFinetuneDataset, self).__init__()
        tm = time.time()
        self.split = split
        self.scale = scale
        self.cfg = cfg
        
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
        
        self.dfs = np.concatenate([np.zeros((self.cfg.wx*self.cfg.window_past, shape1)), self.dfs, np.zeros((self.cfg.wx*self.cfg.window_future, shape1))], axis=0)
        print(f"Dataset initialized in {time.time() - tm} secs!")
        gc.collect()
        
    def read(self, f, _type):
        print(f"Reading {f}...")
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
            row_idx = random.randint(0, self.length-1) + self.cfg.wx*self.cfg.window_past
        elif self.split == "test":
            for i,e in enumerate(self.end_indices):
                if index >= e:
                    continue
                df_idx = i
                break

            row_idx_true = self.shapes[df_idx] - (self.end_indices[df_idx] - index)
            _id = self.f_ids[df_idx] + "_" + str(row_idx_true)
            row_idx = index + self.cfg.wx*self.cfg.window_past
        else:
            row_idx = index + self.cfg.wx*self.cfg.window_past
            
        #scale = 9.806 if self.dfs[row_idx, -1] == 1 else 1.0
        x = self.dfs[row_idx - self.cfg.wx*self.cfg.window_past : row_idx + self.cfg.wx*self.cfg.window_future, 1:4]
        x = x[::self.cfg.wx, :][::-1, :]
        x = torch.tensor(x.astype('float'))#/scale
        
        t = self.dfs[row_idx, -3]*self.dfs[row_idx, -2]
        
        if self.split == "test":
            return _id, x, t
        
        y = self.dfs[row_idx, 4:7].astype('float')
        y = torch.tensor(y)
        
        return x, y, t
    
    def __len__(self):
        return self.length

class FOGPretrainDataset(Dataset):
    def __init__(self, fpaths, scale=9.806, split="train", state="fine-tune"):
        super(FOGPretrainDataset, self).__init__()
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
        
        self.dfs = np.concatenate([np.zeros((self.cfg.wx*self.cfg.window_past, shape1)), self.dfs, np.zeros((2*self.cfg.wx*self.cfg.window_future, shape1))], axis=0)
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
            row_idx = random.randint(0, self.length-1) + self.cfg.wx*self.cfg.window_past
        elif self.split == "test":
            for i,e in enumerate(self.end_indices):
                if index >= e:
                    continue
                df_idx = i
                break

            row_idx_true = self.shapes[df_idx] - (self.end_indices[df_idx] - index)
            _id = self.f_ids[df_idx] + "_" + str(row_idx_true)
            row_idx = index + self.cfg.wx*self.cfg.window_past
        else:
            row_idx = index + self.cfg.wx*self.cfg.window_past

        #scale = 9.806 if self.dfs[row_idx, -1] == 1 else 1.0
        x = self.dfs[row_idx - self.cfg.wx*self.cfg.window_past : row_idx + self.cfg.wx*self.cfg.window_future, 1:4]
        x = x[::self.cfg.wx, :][::-1, :]
        x = torch.tensor(x.astype('float'))
        
        t = self.dfs[row_idx, -3]*self.dfs[row_idx, -2]
        
        y = self.dfs[row_idx + self.cfg.wx*self.cfg.window_future : row_idx + 2*self.cfg.wx*self.cfg.window_future, 1:4]
        y = y[::self.cfg.wx, :][::-1, :]
        y = torch.tensor(y.astype('float'))
        return x, y, t

    def __len__(self):
        return self.length
