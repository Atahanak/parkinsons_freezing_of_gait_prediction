__all__ = ['Config']

import json
import torch

class Config:
    def __init__(self, file_name):
        self.data = {}
        self.file_name = file_name

        KAGGLE = False
        ROOT_READ = '../'
        ROOT_WRITE = '../'
        if KAGGLE:
            ROOT_READ = '/kaggle/input/'
            ROOT_WRITE = '/kaggle/working/'
        self.data["DATA_DIR"] = f'{ROOT_READ}tlvmc-parkinsons-freezing-gait-prediction/'
        self.data["TRAIN_DIR"] = f'{ROOT_READ}tlvmc-parkinsons-freezing-gait-prediction/train/'
        self.data["TDCSFOG_DIR"] = f'{ROOT_READ}tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog/'
        self.data["DEFOG_DIR"] = f'{ROOT_READ}tlvmc-parkinsons-freezing-gait-prediction/train/defog/'
        self.data["CHECKPOINT_PATH"] = f'{ROOT_WRITE}checkpoints/'
        self.data["PARAMS_PATH"] = f'./configs/{self.file_name}.json'

        if KAGGLE:
            self.data["PARAMS_PATH"] = '/' # TODO

        self.data["num_vars"] = 3
        self.data["num_classes"] = 3

        with open(self.data["PARAMS_PATH"]) as f:
            hparams = json.load(f)

        self.data["hparams"] = hparams

        self.data["batch_size"] = hparams["batch_size"]
        self.data["window_size"] = hparams["window_size"]
        self.data["window_future"] = hparams["window_future"]
        self.data["window_past"] = self.data["window_size"] - self.data["window_future"]
        self.data["wx"] = hparams["wx"]

        self.data["model_dropout"] = hparams["model_dropout"]
        self.data["model_hidden"] = hparams["model_hidden"]
        self.data["model_nblocks"] = hparams["model_nblocks"]
        self.data["model_nhead"] = hparams["model_nhead"]

        self.data["lr"] = hparams["lr"]
        self.data["milestones"] = hparams["milestones"]
        self.data["gamma"] = hparams["gamma"]

        self.data["num_epochs"] = hparams["num_epochs"]
        self.data["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data["num_workers"] = 32 if torch.cuda.is_available() else 1

        self.data["feature_list"] = ['AccV', 'AccML', 'AccAP']
        self.data["label_list"] = ['StartHesitation', 'Turn', 'Walking']

        # PatchTST related
        if 'patch_len' in hparams:
            self.data["patch_len"] = hparams["patch_len"]

        self.data["eventsep_model_dropout"] = hparams["model_dropout"]
        self.data["eventsep_model_hidden"] = hparams["model_hidden"]
        self.data["eventsep_model_nblocks"] = hparams["model_nblocks"]
        self.data["eventsep_model_nhead"] = hparams["model_nhead"]

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]
