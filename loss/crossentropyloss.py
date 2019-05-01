import torch.nn as nn


class Loss(object):
    def __init__(self, opt=None):
        super(Loss, self).__init__()

    def __call__(self, *args, **kwargs):
        return nn.CrossEntropyLoss()
