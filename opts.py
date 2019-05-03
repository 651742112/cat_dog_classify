"""参数"""


class opts(object):
    """参数解析"""
    # 实验名称
    exp_name = 'exp1'

    ## 数据集参数设置
    image_size = 64  # 输入时图像大小(备注：对于含全连接层的网络，此处更改后需要更改全连接层的输入通道大小)
    val_ratio = 0.05  # 验证数据集比例
    num_workers = 4  # 加载数据时使用的线程数
    data_name = "{}_{}_{}".format(image_size, image_size, exp_name)  # 数据文件夹名称(备注：可自行更改名称)

    ## 网络模型
    model = 'CNN'  # 可取值'CNN','GoogLeNet'

    ## 优化器
    optimizer = 'SGD'  # 可取值'SDG','RMS','ADAM'

    ## 损失函数
    criterion = 'CrossEntropyLoss'  # 可取值'CrossEntropyLoss'

    ## 训练参数设置
    trainer_name = 'classify'  # 训练器
    batch_size = 100  # 批大小
    num_epochs = 200  # 训练次数
    lr = 0.01  # 学习率
    lr_decay = 0.95  # 衰减率
    weight_decay = 1e-4

    ## 测试模型名称
    test_model_name = "5.pth"  # 待测试模型

    def __init__(self, exp_name='1'):
        super(opts, self).__init__()
