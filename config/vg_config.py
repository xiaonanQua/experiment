import os, glob, json, torch, collections, datetime
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import numpy as np
from pprint import pprint
import torch


class VGConf:
    def __init__(self):
        # 数据集根目录，项目根目录，保存路径
        self.root_path = '/home/team/xiaonan/dataset/visual-gnome/'
        self.project_path = '/home/team/xiaonan/mask-faster-rcnn/'
        self.save_path = '/home/team/xiaonan/datasave/vg_result/'

        # 数据文件:训练、验证、测试文件，字典文件
        self.data_file = {
            'train': self.root_path + 'data/train.txt',
            'val': self.root_path + 'data/val.txt',
            'test': self.root_path + 'data/test.txt',
            'attribute': self.root_path + 'data/attributes_vocab_1000.txt',
            'object': self.root_path + 'data/objects_vocab_2500.txt',
            'relation': self.root_path + 'data/relations_vocab_500.txt',
        }

        # 运行模式:train, train+val, val, mytest
        self.run_mode = 'train+val+test'.split('+')

        # 加载训练、验证、测试数据集
        self.data_list = self.load_data_file()

        # 获得属性、目标、关系字典数据
        self.attr_vocab, self.attr_vocab_idx = \
            self.load_vocab_file(self.data_file['attribute'])
        self.obj_vocab, self.obj_vocab_idx = \
            self.load_vocab_file(self.data_file['object'])
        self.rel_vocab, self.rel_vocab_idx = \
            self.load_vocab_file(self.data_file['relation'])

        # 类别和属性数量
        self.num_classes = len(self.obj_vocab)
        self.num_attributes = len(self.attr_vocab)
        print('object classes number:{}'.format(self.num_classes))
        print('object attribute number:{}'.format(self.num_attributes))

        # 模型名称,检查点,模型路径
        self.model_name = 'faster_rcnn_vgg16'

        # CUDA
        self.device = torch.device('cuda:0')

        # 多项保存目录路径并初始化
        self.mutli_save_path = {
            'log': self.save_path + 'log/' + self.model_name + '/',
            'ckpt': self.save_path + 'cpkts/',
            'result': self.save_path + 'result/result.txt',
            'model': self.save_path + 'cpkts/' + self.model_name + '.pth',
            'pretrained': self.save_path + 'pretrained/vgg16.pth'
        }

        # data
        self.min_size = 600  # image resize
        self.max_size = 1000  # image resize
        self.num_workers = 4
        self.test_num_workers = 4

        # sigma for l1_smooth_loss
        self.rpn_sigma = 3.
        self.roi_sigma = 1.

        # param for optimizer
        # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
        self.weight_decay = 0.0005
        self.lr_decay = 0.1  # 1e-3 -> 1e-4
        self.lr = 1e-3

        # visualization
        self.plot_every = 1  # vis every N iter

        # preset
        self.data = 'voc'
        self.pretrained_model = 'vgg16'

        # training
        self.epoch = 14

        self.use_adam = False  # Use Adam optimizer
        self.use_chainer = False  # try match everything as chainer
        self.use_drop = False  # use dropout in RoIHead

        self.test_num = 10000

        # 预训练模型路径
        self.load_path = None

        self.caffe_pretrain = False  # use caffe pretrained model instead of torchvision
        self.caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

    def init(self):
        """
        初始化文件夹
        Returns:

        """
        for dir_name in ['log', 'ckpt', 'result']:
            if not os.path.exists(self.mutli_save_path[dir_name]):
                os.mkdir(self.mutli_save_path[dir_name])

    def load_data_file(self):
        """
        根据运行模式加载数据文件
        Returns:

        """
        # 定义三个数据列表
        train_list = list()
        val_list = list()
        test_list = list()

        # 定义内嵌函数用于读取txt文件
        def read_data(file_path):
            # 数据
            data = list()

            # 打开文件
            with open(file_path, 'r') as file:
                # 读取每一行
                for data_line in file.readlines():
                    # 将图像路径和注释路径分开
                    ann = data_line.split(' ')
                    # 删除xml路径的‘\n’符
                    ann[1] = ann[1].strip('\n')
                    # 将数据附加到整体数据中
                    data.append(ann)

            return data

        # 根据运行模式获得数据集
        if 'train' in self.run_mode:
            train_list = read_data(self.data_file['train'])
        if 'val' in self.run_mode:
            val_list = read_data(self.data_file['val'])
        if 'mytest' in self.run_mode:
            test_list = read_data(self.data_file['mytest'])

        return (train_list, val_list, test_list)

    def load_vocab_file(self, file_path):
        """
        Reading vocab data from [attributes, relations, objects] txt file
        Args:
            file_path: [attributes, relations, objects] txt file path

            Returns:Vocab Array:['vocab1',...];Vocab to index dict:{'vocab1':index}

        """

        # define vocab data array and vocab into index dict variable
        vocab = []
        vocab_to_idx = {}

        # initial vocab array and dict
        # acquire file name by split function
        file_name = str(file_path.split('/')[-1].split('.')[0])
        vocab = ['__backgrounds__'.format(file_name)
                 if file_name.split('_')[0] in ['objects']
                 else '__no_{}__'.format(file_name.split('_')[0])]
        vocab_to_idx[vocab[0]] = 0

        # reading data from txt file
        with open(file_path, 'r') as file:
            # counting
            count = 1
            # reading by mutli lines
            for data_line in file.readlines():
                name_list = [name.lower().strip() for name in data_line.split(',')]
                vocab.append(name_list[0])

                for name in name_list:
                    vocab_to_idx[name] = count

                count += 1

        return vocab, vocab_to_idx

    def read_image(self, image_file):
        """
        读取图像数据
        Args:
            image_file: 图像的路径

        Returns:

        """
        # 读取图像
        image = Image.open(image_file)

        # 将图像转化成RGB模式和numpy格式
        try:
            img = image.convert('RGB')
            img = np.asarray(img, dtype=np.uint8)
        finally:
            if hasattr(image, 'close'):
                image.close()

        if img.ndim == 2:
            # reshape (H, W) -> (1, H, W)
            return img[np.newaxis]
        else:
            # transpose (H, W, C) -> (C, H, W)
            return img.transpose((2, 0, 1))

    def save_log(self, log_info, write_mode='a+'):
        """
        save log information
        Args:
            log_info: log information
            write_mode: write file mode

        Returns:

        """
        # 打开日志文件
        log = open(self.mutli_save_path['result'], write_mode)
        pre_info = 'Time:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + \
                   '\n' + log_info + '\n'
        log.write(pre_info)
        log.close()

    def parse(self, kwargs):
        # 获得类的属性和值
        state_dict = self._state_dict()

        # 设置参数到类属性配置中
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('unknown option:{}'.format(k))

            # 将参数赋值给属性中
            setattr(self, k, v)

        # 输出配置
        print('----user config----')
        pprint(self._state_dict)

        self.save_log(log_info='----initial config---\n', write_mode='w')

    def _state_dict(self):
        # 获得类的属性值
        return {k: getattr(self, k) for k, _ in VGConf.__dict__.items()
                if not k.startswith('_')}


if __name__ == '__main__':
    cfg = VGConf()
    print(len(cfg.data_list[0]), len(cfg.data_list[1]), len(cfg.data_list[2]))
    print(cfg.attr_vocab, cfg.attr_vocab_idx, '\n',cfg.obj_vocab, cfg.obj_vocab_idx,
          '\n',cfg.rel_vocab, cfg.rel_vocab_idx)

