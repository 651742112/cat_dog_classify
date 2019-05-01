'''训练器定义'''
import time
from torch.autograd import Variable
import torch
from os.path import join
import os

import ref


class Trainer():

    def __init__(self, opt, model, criterion, optimizer):
        # 定义训练需要参数
        self.opt = opt
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, train_loader, val_loader=None):
        # 仅仅为了减少变量长度，不想加self
        opt = self.opt
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer

        # 存放训练、验证的损失和精度
        train_loss = []
        train_accuracy = []
        val_loss = []
        val_accuracy = []

        CUDA = torch.cuda.is_available()
        if CUDA:
            model.cuda()  # 移到GPU上运行

        for epoch in range(opt.num_epochs):
            start = time.time()

            correct = 0
            iter_loss = 0.0  # 累加loss
            model.train()  # 将网络放置训练模式

            for i, (inputs, labels) in enumerate(train_loader):

                if val_loader != None:
                    model.train()  # 将网络放置训练模式

                inputs = Variable(inputs)  # 转为Variable变量
                labels = Variable(labels)

                if CUDA:
                    inputs = inputs.float().cuda()  # 转为float格式并移到GPU
                    labels = labels.long().cuda()

                optimizer.zero_grad()  # 清除上次运行时保存的梯度

                outputs = model(inputs)  # 输入模型得到输出

                loss = criterion(outputs, labels)  # 计算loss （对于一张图像，output是对于的猫和狗的概率，lable为正确的值）
                iter_loss += loss.item()  # 保存loss
                loss.backward()  # 反向传播
                optimizer.step()  # 优化器优化网络参数

                _, predicted = torch.max(outputs, 1)  # 预测结果
                correct += (predicted == labels).sum()  # 累加预测正确的图像个数

            stop = time.time()

            if val_loader != None:
                # 在验证数据集中测试已训练的模型
                val_loss.append(self.val(model, val_loader, criterion)[0])  # 验证数据集中的loss
                val_accuracy.append(self.val(model, val_loader, criterion)[1])  # 验证数据集中的精度
                print('Epoch:{}/{} \t Val Loss: {:.6f} \t Val Acc: {:.2f}% \t Time:{:.3f}s'.format(epoch + 1,
                                                                                                           opt.num_epochs,
                                                                                                           val_loss[
                                                                                                               -1],
                                                                                                           val_accuracy[
                                                                                                               -1],
                                                                                                           stop - start))

            train_loss.append(iter_loss / (len(train_loader) * opt.batch_size))  # 训练数据集中的loss
            train_accuracy.append((int(correct) * 100 / ((len(train_loader) * opt.batch_size))))  # 训练数据集中的精度
            stop = time.time()
            # print('Epoch:{}/{} \t Trainin Loss:{:.6f} \t Training Accuracy:{:.2f}% \t Time:{:.3f}s'.format(epoch + 1,
            #                                                                                                opt.num_epochs,
            #                                                                                                train_loss[
            #                                                                                                    -1],
            #                                                                                                train_accuracy[
            #                                                                                                    -1],
            #                                                                                                stop - start))

            # 保存模型
            save_dir = join(ref.result_root, opt.exp_name)  # 文件夹以参数中的实验名称来命名
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(self.model, save_dir + '/' + str(epoch) + '.pth')
            # 保存日志
            logfile = save_dir + "/{}.log".format(opt.exp_name)
            logStr = "Epoch\tVal Loss\tVal Acc\tTraining Loss\tTraining Acc\tTime\n" if epoch == 0 else ""
            if val_loader != None:
                logStr += "{}\t{:.6f}\t{:.2f}\t{:.6f}\t{:.2f}\n".format(epoch + 1, val_loss[-1], val_accuracy[-1],
                                                                        train_loss[-1],
                                                                        train_accuracy[-1], stop - start)
            else:
                logStr += "{}\t{}\t{}\t{:.6f}\t{:.2f}\n".format(epoch + 1, "None", "None", train_loss[-1],
                                                                train_accuracy[-1], stop - start)
            with open(logfile, 'a') as fn:
                fn.write(logStr)

    def val(self, model, dataloader, criterion):
        model.eval()  # 将模型设置为验证模式

        iter_correct = 0
        iter_loss = 0.0

        for i, (inputs, labels) in enumerate(dataloader):
            inputs = Variable(inputs)
            labels = Variable(labels)

            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.float().cuda()
                labels = labels.long().cuda()

            outputs_ = model(inputs)
            loss = criterion(outputs_, labels)  # Calculate the loss
            iter_loss += loss.item()

            # Record the correct predictions for training data
            _, predicted = torch.max(outputs_, 1)
            iter_correct += (predicted == labels).sum()

        # Record the Testing loss
        val_loss = iter_loss / (len(dataloader) * self.opt.batch_size)
        # Record the Testing accuracy
        val_accuracy = int(iter_correct) * 100 / (len(dataloader) * self.opt.batch_size)

        model.train()
        return val_loss, val_accuracy

    def test(self, test_loader):
        opt = self.opt
        criterion = self.criterion

        model = torch.load(join(ref.result_root, opt.exp_name, opt.test_model_name))
        model.eval()  # Put the network into evaluation mode

        iter_loss = 0.0
        correct = 0.0

        start = time.time()

        for i, (inputs, labels) in enumerate(test_loader):
            # Convert torch tensor to Variable
            inputs = Variable(inputs)
            labels = Variable(labels)

            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.float().cuda()
                labels = labels.long().cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Calculate the loss
            iter_loss += loss.item()
            # Record the correct predictions for training data
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()

        # Record the Testing loss
        test_loss = iter_loss / (len(test_loader) * opt.batch_size)
        # Record the Testing accuracy
        test_accuracy = int(correct) * 100 / (len(test_loader) * opt.batch_size)
        stop = time.time()

        # 保存日志
        save_dir = join(ref.result_root, opt.exp_name)
        logfile = save_dir + "/{}.log".format(opt.exp_name)
        logStr = 'Model:{} \t Testing Loss: {:.6f} \t Testing Acc: {:.2f} \t Total Time: {:.3f}s \t Test Num:{}\n'.format(
            opt.test_model_name,
            test_loss,
            test_accuracy,
            stop - start,
            len(test_loader) * opt.batch_size)
        print(logStr)
        with open(logfile, 'a') as fn:
            fn.write(logStr)
