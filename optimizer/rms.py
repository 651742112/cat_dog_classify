import torch


class Optimizer(object):
    def __init__(self, opt, model):
        super(Optimizer, self).__init__()

        self.opt = opt
        self.model = model

    def __call__(self, *args, **kwargs):
        optimizer = torch.optim.RMSprop(self.model.parameters(),
                                        alpha=0.9,
                                        lr=self.opt.lr,
                                        eps=1e-6,
                                        weight_decay=1e-6
                                        )
        return optimizer
