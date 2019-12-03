from config.config import Config
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder, MNIST
import torchvision.transforms as transforms
import numpy as np
import os


class MnistConfig(Config):
    """
    CatDog数据集的配置文件，继承父类配置文件。
    """

    def __init__(self):
        super(MnistConfig, self).__init__()
        # 图像宽度、高度、通道
        self.image_width = 32
        self.image_height = 32
        self.image_channels = 3
        # 类别数量,类别名称
        self.num_classes = 10
        self.name_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        # 实验的超参数配置
        self.epochs = 50
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.lr_decay_step = 50
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.keep_prob = 0.5

        # 模型的名称
        self.model_name = 'mnist_lenet_v5'
        # 模型检查点地址；日志保存路径
        self.checkpoints = self.model_dir + self.model_name + '.pth'
        self.log_dir = self.log_dir + self.model_name
        if os.path.exists(self.log_dir) is False:
            os.mkdir(self.log_dir)

        # mnist数据集目录、文件名称
        self.mnist_dir = self.root_dataset + 'mnist/'
        self.mnist_train = self.mnist_dir+'train/'

    def dataset_loader(self, root, train=True, shuffle=True, data_preprocess=None, valid_coef=None):
        """
        加载cat_dog数据集
        :param root: 数据存在路径
        :param train: True:则只获得训练集，否则获取测试集
        :param shuffle: True:对小批次数据打乱顺序
        :param data_preprocess: 数据预处理操作,若没有指定，则使用默认的处理操作。若进行验证集的划分，则给出训练集和验证集的字典。
        :param valid_coef: 划分验证集的比例
        :return: 数据集加载器
        """
        dataset = MNIST(root=self.mnist_dir, train=train,
                        transform=data_preprocess, download=True)
        # 获得数据集加载器
        data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle)
        return data_loader



