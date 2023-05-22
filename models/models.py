__all__ = ['FOGModel', 'FOGTransformerEncoder']

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def _block(in_features, out_features, drop_rate):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(drop_rate)
    )

class FOGModel(nn.Module):
    def __init__(self, cfg, state="finetune"):
        super(FOGModel, self).__init__()
        self.cfg = cfg
        p = cfg['model_dropout']
        dim=cfg['model_hidden']
        nblocks=cfg['model_nblocks']
        self.hparams = {}
        self.state = state
        self.dropout = nn.Dropout(p)
        self.in_layer = nn.Linear(cfg['window_size']*3, dim)
        self.blocks = nn.Sequential(*[_block(dim, dim, p) for _ in range(nblocks)])

        self.out_layer_pretrain = nn.Linear(dim, cfg['window_future'] * 3)
        self.out_layer_finetune = nn.Linear(dim, 3)
        
    def forward(self, x):
        x = x.view(-1, self.cfg['window_size']*3)
        x = self.in_layer(x)
        for block in self.blocks:
            x = block(x)
        if self.state == "pretrain":
            x = self.out_layer_pretrain(x)
        else:
            x = self.out_layer_finetune(x)
        return x

class FOGTransformerEncoder(nn.Module):
    def __init__(self, cfg, state="finetune"):
        super(FOGTransformerEncoder, self).__init__()
        self.cfg = cfg
        p = cfg['model_dropout']
        dim=cfg['model_hidden']
        nblocks=cfg['model_nblocks']
        self.hparams = {}
        self.state = state
        self.dropout = nn.Dropout(p)
        self.in_layer = nn.Linear(cfg['window_size']*3, dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=cfg['model_nhead'], dim_feedforward=dim)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=nblocks, mask_check=False)

        self.out_layer_pretrain = nn.Linear(dim, cfg['window_future'] * 3)
        self.out_layer_finetune = nn.Linear(dim, 3)

    def forward(self, x):
        x = x.view(-1, self.cfg['window_size']*3)
        x = self.in_layer(x)
        x = self.dropout(x)
        x = self.transformer(x)
        if self.state == "pretrain":
            x = self.out_layer_pretrain(x)
        else:
            x = self.out_layer_finetune(x)
        return x
