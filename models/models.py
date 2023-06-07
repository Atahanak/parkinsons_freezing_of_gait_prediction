__all__ = ['FOGModel', 'FOGTransformerEncoder', 'FOGEventSeperator', 
           'FOGCNNEventSeperator', 'CNN']

import torch.nn as nn

def _block(in_features, out_features, drop_rate):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(drop_rate)
    )

def _conv_block(in_features, out_features, kernel_size=3, stride=1, padding=1, pool_size=3, pool_stride=2):
    return nn.Sequential(
        nn.Conv1d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
    )
    
class CNN(nn.Module):
    def __init__(self, cfg):
        super(CNN, self).__init__()
        self.cfg = cfg
        p = cfg['model_dropout']
        self.dim=cfg['model_hidden']
        nblocks=cfg['model_nblocks']
        self.padding = 1 #cfg['padding']
        self.dropout = nn.Dropout(p)
        
        # TODO parameterize the number of convolutional layers
        self.conv_layer = _conv_block(cfg['window_size'], self.dim)
        self.MLP = nn.Sequential(*[_block(self.dim, self.dim, p) for _ in range(nblocks)])
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, self.dim)
        for block in self.MLP:
            x = block(x)
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
        self.in_layer = _conv_block(cfg['window_size'], self.dim)
        self.blocks = nn.Sequential(*[_block(self.dim, self.dim, p) for _ in range(nblocks)])
        self.out_layer = nn.Linear(self.dim, 2)
    
    def forward(self, x):
        x = self.in_layer(x)
        # merge the first and third dimension
        x = x.view(-1, self.dim)
        for block in self.blocks:
            x = block(x)
        x = self.out_layer(x)
        return x
