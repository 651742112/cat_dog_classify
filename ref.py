import os
from os.path import join

# 项目根目录
root_dir = os.path.abspath(os.path.dirname(__file__))

# 数据目录
data_root = join(root_dir, 'data')

# 数据集
datasets_root = join(root_dir, 'datasets', 'V1.1')
train_datasets_dir = join(datasets_root, 'train')
test_datasets_dir = join(datasets_root, 'test')

# 结果文件夹
result_root = join(root_dir, 'result')
