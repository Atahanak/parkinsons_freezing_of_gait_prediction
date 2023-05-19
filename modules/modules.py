import pytorch_lightning as pl
import torch
from torchmetrics.functional.classification import multiclass_average_precision
import torch.nn as nn

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
        self.lr = cfg.lr
        # Create model
        self.model = model
        # Create loss module
        self.loss_module = nn.BCEWithLogitsLoss(weight=torch.tensor(loss_weights))
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, cfg.window_size, 3), dtype=torch.float32)
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
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.cfg.gamma)
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
        