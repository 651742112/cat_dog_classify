import torch


class Optimizer(object):
    def __init__(self, opt, model):
        super(Optimizer, self).__init__()

        self.opt = opt
        self.model = model

    def __call__(self, *args, **kwargs):
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.opt.lr,
                                    weight_decay=self.opt.weight_decay
                                    )
        return optimizer
