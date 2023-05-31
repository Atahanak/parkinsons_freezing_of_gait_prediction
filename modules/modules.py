__all__ = ['FOGModule', 'FOGPreTrainModule', 'FOGFinetuneModule']

import pytorch_lightning as pl
import torch
from torchmetrics.functional.classification import multiclass_average_precision
import torch.nn as nn

class FOGFinetuneModule(pl.LightningModule):

    def __init__(self, cfg, model, head, loss_module):
        """
                Generic module for training a model with a head.
        """

        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.lr = cfg['lr']
        self.model = model
        self.head = head
        self.loss_module = loss_module

        self.val_true = None
        self.val_pred = None
    
    def forward(self, x):
        return self.head(self.model(x))
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.milestones, gamma=self.cfg.gamma)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode="min")
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {
          "optimizer": optimizer,
          "lr_scheduler": {
              "scheduler": scheduler,
              "monitor": "val_loss",
              "interval": "epoch",
              "frequency": 1
          }
        }

    def training_step(self, batch, batch_idx):
        x, y, t = batch
        x = x.float()
        #y = torch.tensor([[0, 1] if sum(a) else [1, 0] for a in y]).to(self.cfg['device'])
        y = y.float()
        preds = self.head(self.model(x))
        loss = self.loss_module(preds, y)
        acc = (preds.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('train_acc', acc, sync_dist=True)
        
        with torch.no_grad():
            ap = self.average_precision_score(y, preds)
        
        for i in range(len(ap)):
            self.log(f'train_ap{i}', ap[i], sync_dist=True)
        self.log('train_ap', sum(ap)/len(ap), sync_dist=True)

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        #self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        for i in range(self.head.num_classes):
            avg_precision = self.trainer.logged_metrics[f'train_ap{i}'].nanmean()
            self.log(f'train{i}_precision', avg_precision, sync_dist=True)
        avg_precision = self.trainer.logged_metrics['train_ap'].nanmean()
        self.log('avg_train_precision', avg_precision, sync_dist=True)
    
    def validation_step(self, batch, batch_idx):
        x, y, t = batch
        x = x.float()
        #y = torch.tensor([[0, 1] if sum(a) else [1, 0] for a in y]).to(self.cfg['device'])
        y = y.float()
        preds = self.forward(x)
 
        if self.val_true is None:
            self.val_true = y
            self.val_pred = preds
        else:
            self.val_true = torch.cat((self.val_true, y), dim=0)
            self.val_pred = torch.cat((self.val_pred, preds), dim=0)

        loss = self.loss_module(preds, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return {'loss': loss, 'y_pred': preds, 'y': y}
    
    def on_validation_epoch_end(self):
        acc = (self.val_true.argmax(dim=-1) == self.val_pred.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc, sync_dist=True)
        avg_precision = self.average_precision_score(self.val_true, self.val_pred)
        for i in range(len(avg_precision)):
            self.log(f'val{i}_precision', avg_precision[i], sync_dist=True)
        self.log('avg_val_precision', sum(avg_precision)/len(avg_precision), sync_dist=True)
        self.val_true = None
        self.val_pred = None
    
    def test_step(self, batch, batch_idx):
        x, y, t = batch
        x = x.float()
        #y = torch.tensor([[0, 1] if sum(a) else [1, 0] for a in y]).to(self.cfg['device'])
        y = y.float()
        preds = self.forward(x)
 
        loss = self.loss_module(preds, y)
        if self.val_true is None:
            self.val_true = y
            self.val_pred = preds
        else:
            self.val_true = torch.cat((self.val_true, y), dim=0)
            self.val_pred = torch.cat((self.val_pred, preds), dim=0)
        return {'loss': loss, 'y_pred': preds, 'y': y}
    
    def on_test_epoch_end(self) -> None:
        acc = (self.val_true.argmax(dim=-1) == self.val_pred.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc, sync_dist=True)
        avg_precision = self.average_precision_score(self.val_true, self.val_pred)
        for i in range(len(avg_precision)):
            self.log(f'val{i}_precision', avg_precision[i], sync_dist=True)
        self.log('avg_val_precision', sum(avg_precision)/len(avg_precision), sync_dist=True)
        self.val_true = None
        self.val_pred = None
    
    def average_precision_score(self, y_true, y_pred):
        target = y_true.argmax(dim=-1)
        return multiclass_average_precision(y_pred, target, num_classes=self.head.num_classes, average=None)
class FOGModule(pl.LightningModule):

    def __init__(self, cfg, model, loss_weights, optimizer_name):
        """
        Inputs:
            model_name - Name of the model to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
        """
        super().__init__()
        self.cfg = cfg
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        #self.save_hyperparameters(ignore=["model"])
        self.save_hyperparameters()
        self.lr = cfg['lr']
        # Create model
        self.model = model
        # Create loss module
        self.loss_module = nn.BCEWithLogitsLoss(weight=torch.tensor(loss_weights))
        # Example input for visualizing the graph in Tensorboard
        #self.example_input_array = torch.zeros((1, 1, cfg['window_size'], 3), dtype=torch.float32)
        self.val_true = None
        self.val_pred = None

    def forward(self, past):
        # Forward function that is run when visualizing the graph
        return self.model(past)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.cfg['gamma'])
        else:
            assert False, f"Unknown optimizer: \"{self.hparams.optimizer_name}\""

        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.milestones, gamma=self.cfg.gamma)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode="min")
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {
          "optimizer": optimizer,
          "lr_scheduler": {
              "scheduler": scheduler,
              "monitor": "val_loss",
              "interval": "epoch",
              "frequency": 1
          }
        }
        #return [optimizer], [scheduler]

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
        return {'loss': loss, 'y_pred': preds, 'y': y}
    
    def on_validation_epoch_end(self):
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
 
        loss = self.loss_module(preds, y)
        if self.val_true is None:
            self.val_true = y
            self.val_pred = preds
        else:
            self.val_true = torch.cat((self.val_true, y), dim=0)
            self.val_pred = torch.cat((self.val_pred, preds), dim=0)
        return {'loss': loss, 'y_pred': preds, 'y': y}
    
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
        
class FOGPreTrainModule(pl.LightningModule):
    def __init__(self, cfg, model, optimizer_name, optimizer_hparams):
        super(FOGPreTrainModule, self).__init__()
        self.cfg = cfg
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Example input for visualizing the graph in Tensorboard
        #self.example_input_array = torch.zeros((1, cfg['window_size'], 3), dtype=torch.float32)

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

class FOGEventSeperatorModule(pl.LightningModule):
    def __init__(self, cfg, model, loss_weights, optimizer_name):
        """
        Inputs:
            model_name - Name of the model to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
        """
        super().__init__()
        self.cfg = cfg
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        #self.save_hyperparameters(ignore=["model"])
        self.save_hyperparameters()
        self.lr = cfg['lr']
        # Create model
        self.model = model
        # Create loss module
        self.loss_module = nn.BCEWithLogitsLoss(weight=torch.tensor(loss_weights))
        # # Example input for visualizing the graph in Tensorboard
        # self.example_input_array = torch.zeros((1, cfg['window_size'], 3), dtype=torch.float32)
        self.val_true = None
        self.val_pred = None

    def forward(self, past):
        # Forward function that is run when visualizing the graph
        return self.model(past)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.cfg['gamma'])
        else:
            assert False, f"Unknown optimizer: \"{self.hparams.optimizer_name}\""

        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.milestones, gamma=self.cfg.gamma)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode="min")
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {
          "optimizer": optimizer,
          "lr_scheduler": {
              "scheduler": scheduler,
              "monitor": "val_loss",
              "interval": "epoch",
              "frequency": 1
          }
        }
        #return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        x, y, t = batch
        x = x.float()
        y = torch.tensor([[0, 1] if sum(a) else [1, 0] for a in y]).to(self.cfg['device'])
        y = y.float()
        
        preds = self.model(x)
        loss = self.loss_module(preds, y)
        acc = (preds.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('train_acc', acc, sync_dist=True)
        
        with torch.no_grad():
            ap = self.average_precision_score(y, preds)
        self.log('train_ap0', ap[0], sync_dist=True)
        self.log('train_ap1', ap[1], sync_dist=True)
        self.log('train_ap', sum(ap)/2, sync_dist=True)

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        #self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss  # Return tensor to call ".backward" on

    def on_train_epoch_end(self):
        avg_precision = self.trainer.logged_metrics['train_ap0'].nanmean()
        self.log('train0_precision', avg_precision, sync_dist=True)
        avg_precision = self.trainer.logged_metrics['train_ap1'].nanmean()
        self.log('train1_precision', avg_precision, sync_dist=True)
        self.log('train2_precision', avg_precision, sync_dist=True)
        avg_precision = self.trainer.logged_metrics['train_ap'].nanmean()
        self.log('avg_train_precision', avg_precision, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y, t = batch
        x = x.float()
        y = torch.tensor([[0, 1] if sum(a) else [1, 0] for a in y]).to(self.cfg['device'])
        y = y.float()
        # convert yo to one_hot if y[x] == 1


        preds = self.model(x)
        if self.val_true is None:
            self.val_true = y
            self.val_pred = preds
        else:
            self.val_true = torch.cat((self.val_true, y), dim=0)
            self.val_pred = torch.cat((self.val_pred, preds), dim=0)

        loss = self.loss_module(preds, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return {'loss': loss, 'y_pred': preds, 'y': y}
    
    def on_validation_epoch_end(self):
        acc = (self.val_true.argmax(dim=-1) == self.val_pred.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc, sync_dist=True)
        avg_precision = self.average_precision_score(self.val_true, self.val_pred)
        self.log('val0_precision', avg_precision[0], sync_dist=True)
        self.log('val1_precision', avg_precision[1], sync_dist=True)
        self.log('avg_val_precision', sum(avg_precision)/2, sync_dist=True)
        self.val_true = None
        self.val_pred = None

    def test_step(self, batch, batch_idx):
        x, y, t = batch
        x = x.float()
        y = torch.tensor([[0, 1] if sum(a) else [1, 0] for a in y]).to(self.cfg['device'])
        y = y.float()
        preds = self.model(x)
 
        loss = self.loss_module(preds, y)
        if self.val_true is None:
            self.val_true = y
            self.val_pred = preds
        else:
            self.val_true = torch.cat((self.val_true, y), dim=0)
            self.val_pred = torch.cat((self.val_pred, preds), dim=0)
        return {'loss': loss, 'y_pred': preds, 'y': y}
    
    def on_test_epoch_end(self) -> None:
        acc = (self.val_true.argmax(dim=-1) == self.val_pred.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc, sync_dist=True)
        avg_precision = self.average_precision_score(self.val_true, self.val_pred)
        self.log('val0_precision', avg_precision[0], sync_dist=True)
        self.log('val1_precision', avg_precision[1], sync_dist=True)
        self.log('avg_val_precision', sum(avg_precision)/2, sync_dist=True)
        self.val_true = None
        self.val_pred = None
    
    def average_precision_score(self, y_true, y_pred):
        target = y_true.argmax(dim=-1)
        return multiclass_average_precision(y_pred, target, num_classes=2, average=None)
