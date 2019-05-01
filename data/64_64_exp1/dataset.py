import torch.utils.data as data
from os.path import join
import pickle
import torchvision.transforms as transforms
from PIL import Image

import ref


class Dataset(data.Dataset):
    def __init__(self, opt, split='train'):
        if split == 'train':
            txt = join(ref.data_root, opt.data_name, 'train.txt')
            self.src_dir = join(ref.data_root, opt.data_name, 'train')
        elif split == 'val':
            txt = join(ref.data_root, opt.data_name, 'val.txt')
            self.src_dir = join(ref.data_root, opt.data_name, 'train')
        else:
            txt = join(ref.data_root, opt.data_name, 'test.txt')
            self.src_dir = join(ref.data_root, opt.data_name, 'test')
        self.imgnames = [line.strip('\n') for line in open(txt, 'r')]  # 加载图像名称列表
        # 定义图像转化（此处可定义图像裁剪、缩放、归一化等）
        self.transform = transforms.Compose([
            # transforms.RandomResizedCrop(64),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    def __getitem__(self, index):
        imgname = self.imgnames[index]
        with open(join(self.src_dir, imgname[:-4] + '.pickle'), 'rb') as handle:
            # 打开.pckle的序列化文件读取数据
            output = pickle.load(handle, encoding='latinl')
            image = output['image']
            image = Image.fromarray(image, mode='RGB')  # 将numpy类型的图像转为PIL的图像
            image = self.transform(image)  # 图像预处理
            label = output['label']  # 图像标签
        return image, label

    def __len__(self):
        return len(self.imgnames)
