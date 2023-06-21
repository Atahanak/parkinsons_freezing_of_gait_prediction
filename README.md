# Parkinson's Freezing of Gait Prediction

This repository contains models and datasets developed for multi-variate time-series data. The codes are specifically written for the Kaggle competition [Parkinson's Freezing of Gait Prediction](https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/overview).

This repository implements MLP, CNN and Transformer-Encoder models. All of these models can be utilized with the scripts described below. Complete models that include the model body and the head are provided with `FOG` prefix.

Also one of the recent works on time series data with transformers, [PatchTST](https://github.com/yuqinie98/PatchTST), is also integrated into the repository.

You can find several notebooks for detailed data anaylsis under `data_analysis/`. I would like to express my gratitude to fellow Kagglers for providing some of these notebooks.

You can use the scripts as:

- **pretrain.py**: Employs self-supervised learning. Learns to predict future readings from previous data. The competition data contained a huge unlabeled dataset as well which is perfect for pretraining.

- **finetune.py**: Employs supervised learning. Finetunes any given model, it can also be a pretrained model, with the labeled data (defog, tdcsfog).

- **separate_event.py**: Employs supervised learning. This script learns if there is an event (Walking,  Turning, Start of Hesitation) or not. In addition to Defog and Tdcsfog, Notype data can also be used.

- **separate_predict.py**: Employs supervised learning. This script is the combination of `finetune.py` and `seperate_event.py`.
### Configuration

All parameters of the models and data are stored in config.py and initialized using a .json file. Here's an example file:

```json
{
    "batch_size": 1024,
    "window_size": 32,
    "window_future": 8,
    "wx": 1,

    "model_dropout": 0.2,
    "model_hidden": 512,
    "model_nblocks": 1,
    "model_nhead": 8,

    "lr": 0.00015,
    "milestones": [5, 10, 15, 20],
    "gamma": 0.00003,
    "num_epochs": 2,

    "use_pretrained": false,

    "patch_len": 4,

    "eventsep_model_dropout": 0.2,
    "eventsep_model_hidden": 512,
    "eventsep_model_nblocks": 1,
    "eventsep_model_nhead": 8,

    "num_classes": 3
}
```
This file can be used for all the scripts including pretraining.

### Train-Validation-Test Split

Stratified Group K fold with respect to each patient is used for the train, validation split. This preserves the class distribution of `Start of Hesitation`, `Walking`, and `Turn` classes, while including data samples of patients both in the train and validation set. Various different splits are observed and `i=2` is choosen as the best split. Test set is omitted by Kaggle and can only be used during submission.

Stratified Group K-fold with respect to each patient is used for the train-validation split. This approach preserves the class distribution of `Start of Hesitation`, `Walking`, and `Turn` classes while including data samples from patients in both the train and validation sets. Among various different splits, `i=2` has been chosen as the best split. The test set is omitted by Kaggle and can only be used during submission.

### Models

I have experimented with four different models for multi-variate time-series classification:

**MLP**

Multi-layer perceptron is the simplest model among the four. This model employs `model_nblocks` number of layers. Each layer consists of a linear, batch normalization, ReLu and dropout function in that order.

**CNN**

CNN model adds a `_conv_block` on top of the MLP model. `_conv_block` contains 1D convolution, ReLU and Max Pool 1D respectively.

**Transformer-Encoder**

As shown by [BERT](https://arxiv.org/abs/1810.04805), for classification, transformer encoders display stronger results than the decoder only or mixed architectures. For the transformer encoder I used PyTorch's native `pytorch.nn.TransformerEncoder` class.

**PatchTST**

[PatchTST](https://github.com/yuqinie98/PatchTST) is a state of the art transformer based model for multi-variate time-series analysis that is published in [ICLR 2023](https://iclr.cc/). The main takeaways from the paper are: 

- Patching: time-series data is patched, i.e. divided into predefined length of segments (patch length), these so called patches are analogous to tokens in a natural language text.
- Channel-Independence: Each channel, i.e. data column or sensor, shares the same embedding and Transformer weights.

For more details please refer to the original [paper](https://arxiv.org/abs/2211.14730).

### Training Details & PyTorch Lightning

PyTorch Lightning is a valuable tool for simplifying training and accelerating development in PyTorch-based projects. Acting as a lightweight and modular wrapper over PyTorch, it reduces the need for boilerplate code and promotes the use of best practices.

In this project, I utilized PyTorch Lightning for the first time, and to be honest, I wouldn't have made it this far without it. It significantly accelerated my development-test-evaluate cycle, allowing me to focus on designing and implementing models rather than dealing with low-level details.

**Loss Function**

For the loss function I employed binary cross entropy with logits loss, `nn.BCEWithLogitsLoss`. Since the number of classes are not balanced trough the data, a weight vector that is inversely proportional to the number of positive samples for each class is used. This practice is also carried out for the `event_seperator` models.

**Optimizer & Schedular**

For the optimizer I experimented with `SGD` and `Adam` optimizer and found the latter outperforms the former. Additionally, I also experimented with different learning rate scheduling techniques, including the vanilla scheduler, `MultiStepLR`, `ReduceLROnPlateau`, and `CosineAnnealingLR`. Out of these four options, `ReduceLROnPlateau` yielded the best loss scores.


**Automatic Tuning**

Due to time constraints, besides `batch_size` and `learning_rate`, no other hyper parameter search is carried out. For the latter PyTorch Lightning's `tuner.scale_batch_size` (binary search) and  `tuner.lr_find` features are used.



