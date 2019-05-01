'''生成训练、验证、测试数据'''
import os
from joblib import Parallel, delayed
import cv2
import copy
import pickle

import ref
from opts import opts


def gen_train_data():
    '''生成训练数据'''
    opt = opts()
    data_save_dir = os.path.join(ref.data_root, opt.data_name, 'train')
    init_folder([data_save_dir])  # 初始化数据保存目录

    '''(1)step1: 并行加载数据集，然后保存增强过的数据'''
    Parallel(n_jobs=opt.num_workers)(
        delayed(load_and_save)(opt, f_name, ref.train_datasets_dir, data_save_dir, split='train') for f_name in
        os.listdir(ref.train_datasets_dir))

    '''(2)step2:保存训练、验证数据的name'''
    train_name = os.path.join(ref.data_root, opt.data_name, 'train.txt')
    val_name = os.path.join(ref.data_root, opt.data_name, "val.txt")
    dirname = data_save_dir

    def valid_pickle_file(x):
        return x.endswith('.jpg') and os.path.isfile("{}/{}.pickle".format(dirname, x[:-4]))

    imgnames = [x for x in os.listdir(dirname) if valid_pickle_file(x)]
    val_num = int((len(imgnames) * opt.val_ratio) / 2)
    train = imgnames[val_num:len(imgnames) - val_num]
    val = imgnames[:val_num] + imgnames[len(imgnames) - val_num:]

    def write_list_to_file(f_, filelist):
        with open(f_, 'w') as h:
            for l in filelist:
                h.write("{}\n".format(l))

    write_list_to_file(train_name, train)
    write_list_to_file(val_name, val)


def gen_test_data():
    '''生成测试数据'''
    opt = opts()  # 参数
    data_save_dir = os.path.join(ref.data_root, opt.data_name, 'test')
    init_folder([data_save_dir])  # 初始化目录

    '''(1)step1: 并行加载数据集，然后保存数据'''
    Parallel(n_jobs=opt.num_workers)(
        delayed(load_and_save)(opt, f_name, ref.test_datasets_dir, data_save_dir, split='test') for f_name in
        os.listdir(ref.test_datasets_dir))  # 并行加载图像处理并保存

    '''(2)step2:保存测试数据的name'''
    test_name = os.path.join(ref.data_root, opt.data_name, 'test.txt')
    dirname = data_save_dir

    def valid_pickle_file(x):
        return x.endswith('.jpg') and os.path.isfile("{}/{}.pickle".format(dirname, x[:-4]))

    imgnames = [x for x in os.listdir(dirname) if valid_pickle_file(x)]
    test = imgnames

    def write_list_to_file(f_, filelist):
        with open(f_, 'w') as h:
            for l in filelist:
                h.write("{}\n".format(l))

    write_list_to_file(test_name, test)


def init_folder(dir_list):
    '''初始化目录'''
    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)


def load_and_save(opt, file_name, src_dir, save_dir, split='train'):
    d = loading(file_name, src_dir)  # 根据file_name和src_dir拼接为图片路径，读取图片

    dlist = []  # 保存图片和由它增强的其他的图片
    dlist.append(d)

    # 训练数据增强
    if split == 'train':
        dlist.append(make_mirror(d, axis=2))  # 水平旋转

    outputs = [resize_and_transform(opt, t) for t in dlist]  # 图片缩放

    ## 保存所有图片
    ## 以.pickle的序列化文件保存输入数据和标签
    for fn in outputs:
        with open('{}/{}.pickle'.format(save_dir, fn['imgname'][:-4]), 'wb') as handle:
            pickle.dump(fn, handle, protocol=pickle.HIGHEST_PROTOCOL)  # @f
        cv2.imwrite(os.path.join(save_dir, fn['imgname']), fn['image'])


def loading(f_name, src_dir):
    '''加载图片'''
    f_path = os.path.join(src_dir, f_name)
    d = {}
    d['image'] = cv2.imread(f_path)
    label, name, suffix = f_name.split('.')
    d['label'] = 0 if label == 'cat' else 1
    d['imgname'] = '{}{}.{}'.format(label, name, suffix)
    return d


def make_mirror(d, axis=1):
    '''图像仿射变换'''
    d_mirrorr = copy.deepcopy(d)
    I = d_mirrorr['image']

    if axis == 2:
        # 水平旋转
        d_mirrorr['image'] = I[:, ::-1, :]
        name, suffix = d_mirrorr['imgname'].split('.')
        d_mirrorr['imgname'] = "{}_mr.{}".format(name, suffix)
    else:
        # 垂直旋转
        d_mirrorr['image'] = I[::-1, :, :]
        name, suffix = d_mirrorr['imgname'].split('.')
        d_mirrorr['imgname'] = "{}_ud.{}".format(name, suffix)
    return d_mirrorr


def resize_and_transform(opt, d):
    '''使用opencv缩放图像'''
    inp_h = inp_w = opt.image_size
    img = d['image']

    try:
        resized_img = cv2.resize(img, (inp_h, inp_w), cv2.INTER_CUBIC)
    except ValueError:
        print("cannot resize ", d['imgname'])
        raise ValueError

    d['image'] = resized_img
    return d


if __name__ == '__main__':
    gen_train_data()
    gen_test_data()
