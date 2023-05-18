import pytorch_lightning as pl
import torchvision
import torchmetrics

class ConfusionMatrixCallback(pl.Callback):
    def __init__(self, cfg, num_classes, task="multiclass"):
        super().__init__()
        self.cfg = cfg
        self.conf_matrix = torchmetrics.ConfusionMatrix(num_classes=num_classes, task=task).to(cfg.device)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        # Get the predicted labels and ground truth labels from the batch
        y_pred, y_true = outputs['y_pred'].argmax(dim=-1), outputs['y'].argmax(dim=-1)

        # Update the confusion matrix with the current batch
        self.conf_matrix.update(y_pred, y_true)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Compute the confusion matrix for the entire validation set
        matrix = self.conf_matrix.compute().detach().cpu()

        # Print the confusion matrix
        print('Confusion matrix:')
        print(matrix)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        # Get the predicted labels and ground truth labels from the batch
        y_pred, y_true = outputs['y_pred'].argmax(dim=-1), outputs['y'].argmax(dim=-1)

        # Update the confusion matrix with the current batch
        self.conf_matrix.update(y_pred, y_true)

    def on_test_end(self, trainer, pl_module):
        # Compute the confusion matrix for the entire test set
        matrix = self.conf_matrix.compute().detach().cpu()
        #convert the confusion matrix to a tensor and add it to TensorBoard
        #conf_matrix = torch.from_numpy(matrix).float()
        conf_matrix = torchvision.utils.make_grid(matrix.float().unsqueeze(0), normalize=True, scale_each=True)
        trainer.logger.experiment.add_image('Confusion matrix', conf_matrix)

        # Print the confusion matrix
        print('Confusion matrix:')
        print(matrix)
