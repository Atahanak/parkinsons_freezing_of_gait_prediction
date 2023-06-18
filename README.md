# Parkinson's Freezing of GAIT Prediction

This repository contains models and datasets developed for multi-variate time-series data. 

An MLP, CNN and Transformer-Encoder is implemented. All of these models can be utilized with the scripts described below. 

Also one of the recent works on time series data with transformers, [PatchTST](https://github.com/yuqinie98/PatchTST), is also integrated to the repository.

You can use the scripts as:

- **pretrain.py**: Employs self-supervised learning. Learns to predict future readings from previous data. I used the unlabeled dataset from the competition for pretraining.

- **finetune.py**: Employs supervised learning. Finetunes any given model, it can also be a pretrained model, with the labeled data (defog, tdcsfog).

- **separate_event.py**: Employs supervised learning. This script learns if there is an event (Walking,  Turning, Start of Hesitation) or not. In addition to Defog and Tdcsfog, Notype data can also be used.

- **separate_predict.py**: Employs supervised learning. This script is the combination of `finetune.py` and `seperate_event.py`.

