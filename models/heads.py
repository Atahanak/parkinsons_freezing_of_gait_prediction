__all__ = ['SeperatorHead', 'ClassifierHead', 'ForecasterHead']

import torch.nn as nn

class ForecasterHead(nn.Module):
    def __init__(self, num_future, dim):
        super(ForecasterHead, self).__init__()
        self.out_layer = nn.Linear(dim, num_future)

    def forward(self, x):
        x = self.out_layer(x)
        return x
    
class ClassifierHead(nn.Module):
    def __init__(self, num_classes, dim):
        super(ClassifierHead, self).__init__()
        self.num_classes = num_classes
        self.out_layer = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.out_layer(x)
        return x

class SeperatorHead(nn.Module):
    def __init__(self, dim):
        super(SeperatorHead, self).__init__()
        self.num_classes = 2
        self.out_layer = nn.Linear(dim, self.num_classes)

    def forward(self, x):
        x = self.out_layer(x)
        return x
    