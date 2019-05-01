import torch


class Optimizer(object):
    def __init__(self, opt, model):
        super(Optimizer, self).__init__()

        self.opt = opt
        self.model = model

    def __call__(self, *args, **kwargs):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.opt.lr, betas=(0.9, 0.999),
                                     eps=1e-8,
                                     weight_decay=1e-6
                                     )
        return optimizer
