__all__ = ['FOGModel', 'FOGTransformerEncoder', 'FOGEventSeperator', 'FOGCNNEventSeperator']

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

class FOGForecastHead(nn.Module):
    def __init__(self, cfg, dim):
        super(FOGForecastHead, self).__init__()
        self.out_layer = nn.Linear(dim, cfg['window_future'] * 3)

    def forward(self, x):
        x = self.out_layer(x)
        return x
    
class FOGClassifierHead(nn.Module):
    def __init__(self, cfg, dim):
        super(FOGClassifierHead, self).__init__()
        self.out_layer = nn.Linear(dim, 3)

    def forward(self, x):
        x = self.out_layer(x)
        return x

class FOGSeperatorHead(nn.Module):
    def __init__(self, cfg, dim):
        super(FOGSeperatorHead, self).__init__()
        self.out_layer = nn.Linear(dim, 2)

    def forward(self, x):
        x = self.out_layer(x)
        return x

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

class FOGEventSeperator(nn.Module):
    def __init__(self, cfg):
        super(FOGEventSeperator, self).__init__()
        self.cfg = cfg
        p = cfg['eventsep_model_dropout']
        dim=cfg['eventsep_model_hidden']
        nblocks=cfg['eventsep_model_nblocks']
        self.hparams = {}
        self.dropout = nn.Dropout(p)
        self.in_layer = nn.Linear(cfg['window_size']*3, dim)
        self.blocks = nn.Sequential(*[_block(dim, dim, p) for _ in range(nblocks)])
        self.out_layer = nn.Linear(dim, 2)
    
    def forward(self, x):
        x = x.view(-1, self.cfg['window_size']*3)
        x = self.in_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_layer(x)
        return x

class FOGCNNEventSeperator(nn.Module):
    def __init__(self, cfg):
        super(FOGCNNEventSeperator, self).__init__()
        self.cfg = cfg
        p = cfg['eventsep_model_dropout']
        self.dim=cfg['eventsep_model_hidden']
        nblocks=cfg['eventsep_model_nblocks']
        self.hparams = {}
        self.padding = 1
        self.dropout = nn.Dropout(p)
        #self.in_layer = nn.Linear(cfg['window_size']*3, dim)
        self.in_layer = self.conv_block(cfg['window_size'], self.dim)
        self.blocks = nn.Sequential(*[_block(self.dim, self.dim, p) for _ in range(nblocks)])
        self.out_layer = nn.Linear(self.dim, 2)

    def conv_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Conv1d(in_features, out_features, kernel_size=3, stride=1, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
    
    def forward(self, x):
        x = self.in_layer(x)
        # merge the first and third dimension
        x = x.view(-1, self.dim)
        for block in self.blocks:
            x = block(x)
        x = self.out_layer(x)
        return x
