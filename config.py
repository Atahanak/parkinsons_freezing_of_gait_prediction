__all__ = ['Config']

import json
import torch

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

    num_vars = 3
    num_classes = 3

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

    # PatchTST related
    patch_len = hparams["patch_len"]