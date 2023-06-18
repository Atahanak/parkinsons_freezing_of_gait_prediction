#!/usr/bin/env python
# coding: utf-8

import os
import time
import random
import gc
import glob
import json

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import MixedPrecisionPlugin 
import torchmetrics
torch.set_float32_matmul_precision('high')

from watermark import watermark
print(watermark(packages="pytorch_lightning, torchmetrics, torch, sklearn, pandas, numpy"))

from utils import *

if __name__ == '__main__':
    # In[165]:
    from config import Config
    cfg = Config(file_name='config')
    #pretty print the config
    print(json.dumps(cfg.data, indent=4, sort_keys=True))

    #loss_weights = [0.5625255840823298, 0.4374744159176702]
    #loss_weights = [0.4374744159176702, 0.5625255840823298]
    # total_td , event_total_td = event_analysis(cfg, 'tdcsfog')
    total_de , event_total_de = event_analysis(cfg, 'defog')
    # total_no , event_total_no = event_analysis(cfg, 'notype')

    # total = total_td + total_de + total_no
    total = total_de
    # event_total = event_total_td + event_total_de + event_total_no
    event_total = event_total_de

    # #total = 10251114
    # #event_total = 6832879
    print(f"Total: {total}, number of events: {event_total}")

    # #get loss weights using total no and event total
    loss_weights = [(total - event_total) / total, event_total / total ]

    print(f"Loss weights: {loss_weights}")

    from utils import *
    #split_analysis(cfg, 'tdcsfog')
    train_fpaths_tdcs, valid_fpaths_tdcs = split_data(cfg, 'tdcsfog', 2)
    #split_analysis(cfg, 'defog')
    train_fpaths_de, valid_fpaths_de = split_data(cfg, 'defog', 2)
    train_fpaths_no = glob.glob(f"{cfg['DATA_DIR']}/train/notype/*")
    #train_fpaths = [(f, 'de') for f in train_fpaths_de] + [(f, 'tdcs') for f in train_fpaths_tdcs] 
    train_fpaths = [(f, 'tdcs') for f in train_fpaths_tdcs] 
    #train_fpaths = [(f, 'de') for f in train_fpaths_de] 
    train_fpaths_2 = [(f, 'notype') for f in train_fpaths_no]
    # train_fpaths = [(f, 'notype') for f in train_fpaths_no]
    #valid_fpaths = [(f, 'de') for f in valid_fpaths_de] + [(f, 'tdcs') for f in valid_fpaths_tdcs]
    valid_fpaths = [(f, 'tdcs') for f in train_fpaths_tdcs] 
    #valid_fpaths = [(f, 'de') for f in valid_fpaths_de] 

    gc.collect()

    from dataset.Dataset import FOGDataset
    fog_train = FOGDataset(train_fpaths, cfg, task = 'seperate')
    fog_train_2 = FOGDataset(train_fpaths_2, cfg, task = 'seperate')
    fog_valid = FOGDataset(valid_fpaths, cfg, task = 'seperate')

    # load no label dataset

    print("TRAIN")
    print("Dataset size:", fog_train.__len__())
    print("Dataset size:", fog_train_2.__len__())
    print("VALID")
    print("Dataset size:", fog_valid.__len__())

    from models.models import MLP  as Seperator
    model = Seperator(cfg)
    from models.heads import SeperatorHead
    head = SeperatorHead(cfg['model_hidden'])
    print(f'The model has {count_parameters(model):,} trainable parameters')

    from modules.modules import FOGFinetuneModule
    from modules.callbacks import ConfusionMatrixCallback
    from pytorch_lightning.utilities import CombinedLoader

    def train_model(module, model, head, train_dataset, val_dataset, save_name = None, **kwargs):
        """
        Inputs:
            model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
            save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
        """
        # Create a PyTorch Lightning trainer with the generation callback
        conf_matrix_callback = ConfusionMatrixCallback(cfg, num_classes=head.num_classes, task="multiclass")
        #combined_train_loaders = CombinedLoader(train_loaders, mode="sequential")
        trainer = pl.Trainer(default_root_dir=os.path.join(cfg['CHECKPOINT_PATH'], save_name),                          # Where to save models
                            plugins=[MixedPrecisionPlugin(precision="bf16-mixed", device=cfg['device'])],
                            accelerator="gpu" if str(cfg['device']).startswith("cuda") else "cpu",                     # We run on a GPU (if possible)
                            devices=torch.cuda.device_count() if str(cfg['device']).startswith("cuda") else 1,         # How many GPUs/CPUs we want to use (1 is enough for the notebooks)
                            max_epochs=cfg['num_epochs'],                                                                     # How many epochs to train for if no patience is set
                            callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="avg_val_precision"),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                            conf_matrix_callback,
                                        LearningRateMonitor("epoch")],                                           # Log learning rate every epoch
                            enable_progress_bar=True,                                                          # Set to False if you do not want a progress bar
                            logger = True,
                            #strategy=DDPStrategy(find_unused_parameters=True),
                            #val_check_interval=0.5,
                            log_every_n_steps=50)                                                           
        trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
        trainer.logger._default_hp_metric = True

        # log hyperparameters, including model and custom parameters
        if 'milestones' in cfg['hparams']:
            del cfg['hparams']["milestones"] 
        trainer.logger.log_metrics(cfg["hparams"])

        # Check whether pretrained model exists. If yes, load it and skip training
        pretrained_filename = os.path.join(cfg["CHECKPOINT_PATH"], save_name + f"/{cfg['model_hidden']}_{cfg['model_nblocks']}_final.pt")
        print(f"Pretrained file {pretrained_filename}")
        if cfg['hparams']["use_pretrained"] and os.path.isfile(pretrained_filename):
            print(f"Found pretrained model at {pretrained_filename}, loading...")
            model.load_state_dict(torch.load(pretrained_filename)) # Automatically loads the model with the saved hyperparameters
        loss_module = nn.BCEWithLogitsLoss(weight=torch.tensor(loss_weights))
        lmodel = module(cfg, model, head, loss_module, train_dataset, val_dataset)

        # tune learning rate
        print("Tuning learning rate...")
        tuner = Tuner(trainer)
        # Run learning rate finder
        lr_finder = tuner.lr_find(lmodel) #, train_dataloaders=train_loaders[0], val_dataloaders=val_loader)
        # Auto-scale batch size with binary search
        #tuner.scale_batch_size(lmodel, mode="binsearch")
        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        print(f"New learning rate: {new_lr}")
        print("Tuning done.")
        
        pl.seed_everything(42) # To be reproducable
        trainer.fit(lmodel) #, train_loaders[0], val_loader)
        #trainer.fit(lmodel, train_loaders[1], val_loader)
        print(f"Best model path {trainer.checkpoint_callback.best_model_path}")
        lmodel = module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

        # Test best model on validation set
        val_result = trainer.test(lmodel, verbose=False)
        return lmodel, trainer, val_result

    seperator_model, trainer_seperator, val_result = train_model(FOGFinetuneModule, model, head, fog_train, fog_valid, save_name="FOGEventSeperator", optimizer_name="Adam")
    result = {
                    "val_ap": val_result[0]["avg_val_precision"],
                    "no": val_result[0]["val0_precision"],
                    "yes": val_result[0]["val1_precision"],
                }
    print(json.dumps(cfg['hparams'], sort_keys=True, indent=4))
    print(json.dumps(result, sort_keys=True, indent=4))

    print('Total length:', fog_train.__len__())
    print('Total length:', fog_valid.__len__())
    fog_train.task = 'classify'
    fog_valid.task = 'classify'
    print('Total length:', fog_train.__len__())
    print('Total length:', fog_valid.__len__())

    with open(f"{cfg['DATA_DIR']}/events.csv") as file:
        df = pd.read_csv(file)
        # create a new dataframe column by take the difference between begin and end columns
        df['duration'] = df['Completion'] - df['Init']
        # take the mean of the duration column
        mean = df['duration'].mean()
        print(df)
        # create a new dataframe by grouping each task according to their mean duration and also count 
        df = df.groupby('Type').agg({'duration': ['sum', 'count']})
        # sort dt by mean duration
        df = df.sort_values(by=[('duration', 'sum')], ascending=False)
        # get numpy array of duration sum inversely scaled between 0 and 1
        counts = df[('duration', 'sum')].values
        #loss_weights = 1 - (counts / counts.sum()) # inverse scale
        loss_weights = 1 / counts # inverse scale
        loss_weights = loss_weights / loss_weights.sum() * 3
        print("Loss weights: ", loss_weights, loss_weights.sum())

    from models.models import TransformerEncoder as Learner
    model = Learner(cfg)
    from models.heads import ClassifierHead
    head = ClassifierHead(cfg['num_classes'], cfg['model_hidden'])
    classifier_model, trainer_classifier, val_result = train_model(FOGFinetuneModule, model, head, fog_valid, fog_train, save_name="FOGEventSeperator", optimizer_name="Adam")
    print(json.dumps(cfg['hparams'], sort_keys=True, indent=4))
    result = {
            "val_ap": val_result[0]["avg_val_precision"],
            "SH": val_result[0]["val0_precision"],
            "T": val_result[0]["val1_precision"],
            "W": val_result[0]["val2_precision"],
        }
    print(json.dumps(result, sort_keys=True, indent=4))

    # # ## Submission
    seperator = FOGFinetuneModule.load_from_checkpoint(trainer_seperator.checkpoint_callback.best_model_path)
    seperator.to(cfg['device'])
    seperator.eval()

    classifier = FOGFinetuneModule.load_from_checkpoint(trainer_classifier.checkpoint_callback.best_model_path)
    classifier.to(cfg['device'])
    classifier.eval()

    test_defog_paths = glob.glob(f"{cfg['DATA_DIR']}test/defog/*.csv")
    test_tdcsfog_paths = glob.glob(f"{cfg['DATA_DIR']}test/tdcsfog/*.csv")
    test_fpaths = [(f, 'de') for f in test_defog_paths] + [(f, 'tdcs') for f in test_tdcsfog_paths]
    print(test_fpaths)
    test_dataset = FOGDataset(test_fpaths, cfg, split="test")
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size']) #, num_workers=cfg['num_workers'])
    print('test_loader', next(iter(test_loader))[0].shape)
    ids = []
    preds = []

    for _id, x, _ in test_loader: #tqdm(test_loader):
        x = x.to(cfg['device']).float()
        with torch.no_grad():
            print(x.shape)
            y_sep = seperator(x)
            y_pred = classifier(x)
            for i, y in enumerate(y_sep):
                print(y)
                if y[0] > y[1]:
                    y_pred[i] = torch.tensor([0, 0, 0])

        ids.extend(_id)
        preds.extend(list(np.nan_to_num(y_pred.cpu().numpy())))

    sample_submission = pd.read_csv(f"{cfg['DATA_DIR']}sample_submission.csv")
    print(sample_submission.shape)

    preds = np.array(preds)
    print(preds.shape)
    print(preds)
    submission = pd.DataFrame({'Id': ids, 'StartHesitation': np.round(preds[:,0],5), \
                               'Turn': np.round(preds[:,1],5), 'Walking': np.round(preds[:,2],5)})

    submission = pd.merge(sample_submission[['Id']], submission, how='left', on='Id').fillna(0.0)
    submission.to_csv(f"submission.csv", index=False)

    print(submission.shape)
    submission.head()

