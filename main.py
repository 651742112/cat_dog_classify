import torch
import importlib

from opts import opts


def init_model(opt):
    '''动态导入模型'''
    modelLib = importlib.import_module('model.{}'.format(opt.model.lower()))
    return getattr(modelLib, opt.model)()


def init_criterion(opt):
    criterionLib = importlib.import_module('loss.{}'.format(opt.criterion.lower()))
    return criterionLib.Loss(opt)()


def init_optimizer(opt, model):
    optimizerLib = importlib.import_module('optimizer.{}'.format(opt.optimizer.lower()))
    return optimizerLib.Optimizer(opt, model)()


def init_trainer(opt, model, criterion, optimizer):
    trainerLib = importlib.import_module('trainer.{}'.format(opt.trainer_name))
    return trainerLib.Trainer(opt, model, criterion, optimizer)


def init_dataloader(opt, split='train'):
    dataste = importlib.import_module('data.{}.dataset'.format(opt.data_name))  # 动态导入数据集
    loader = None
    if split == 'train':
        loader = torch.utils.data.DataLoader(
            dataste.Dataset(opt, split=split),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True)
    elif split == 'val':
        loader = torch.utils.data.DataLoader(
            dataste.Dataset(opt, split=split),
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(
            dataste.Dataset(opt, split=split),
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True)
    return loader


def train():
    opt = opts()
    model = init_model(opt)  # 定义网络
    criterion = init_criterion(opt)  # 定义损失函数
    optimizer = init_optimizer(opt, model)  # 定义优化器
    trainer = init_trainer(opt, model, criterion, optimizer)  # 定义训练器
    train_loader = init_dataloader(opt, 'train')  # 加载训练数据
    val_loader = init_dataloader(opt, 'val')  # 加载验证数据
    trainer.train(train_loader, val_loader)  # 训练


def test_model():
    opt = opts()
    model = init_model(opt)  # 定义网络
    criterion = init_criterion(opt)  # 定义损失函数
    optimizer = init_optimizer(opt, model)  # 定义优化器
    trainer = init_trainer(opt, model, criterion, optimizer)  # 定义训练器
    test_loader = init_dataloader(opt, 'test')  # 加载测试数据
    trainer.test(test_loader)  # 测试一个网络模型，模型在opt参数里面定义

    # 如若需要测试所有模型，请如下遍历所有模型
    # for i in range(130):
    #     trainer.opt.test_model_name = "{}.pth".format(i) #模型名称
    #     trainer.test(test_loader)  # 测试


if __name__ == '__main__':
    train()
    # test_model()
