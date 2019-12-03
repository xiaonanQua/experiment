from config.cifar10_config import Cifar10Config
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder
import torchvision.transforms as transforms


class TestConfig(Cifar10Config):
    """
    配置代码测试数据集
    """

    def __init__(self):
        super(TestConfig, self).__init__()
        # 图像宽度、高度、通道
        self.image_width = 32
        self.image_height = 32
        self.image_channels = 3
        # 类别数量
        self.num_classes = 2
        # 实验的超参数配置
        self.epochs = 100
        self.batch_size = 32
        self.learning_rate = 0.1  # 原始是0.01
        self.learning_rate_decay = 0.95
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.keep_prob = 0.5

        # 模型的名称
        self.model_name = 'catdog'
        # 模型检查点地址
        self.checkpoints = self.model_dir + self.model_name + '.pth'
        # 梯度累积
        self.grad_accuml = True
        self.batch_accumulate_size = 4

        # 训练集、测试集预处理操作
        self.train_preprocess = transforms.Compose([
            transforms.Resize(size=(self.image_height, self.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.test_preprocess = transforms.Compose([
            transforms.Resize(size=(self.image_height, self.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # 代码测试数据集目录
        self.test_dir = self.root_dataset + 'test/'

    def dataset_loader_test(self, root, shuffle=True, data_preprocess=None):
        """
        加载代码测试数据集
        :param root: 数据存在路径
        :param train: True:则只获得训练集，否则获取测试集
        :param shuffle: True:对小批次数据打乱顺序
        :param data_preprocess: 数据预处理操作,若没有指定，则使用默认的处理操作
        :return: 数据集加载器
        """

        # 若预处理为空，则使用默认的
        if data_preprocess is None:
            data_preprocess = self.train_preprocess

        # 获得训练数据集
        # dataset = CIFAR10(root=root, train=train, transform=data_preprocess, download=True)
        dataset = ImageFolder(root=root, transform=data_preprocess)
        print(dataset.imgs)
        # 类别名称
        self.dict_classes = dataset.class_to_idx
        self.classes = dataset.classes
        print('类别：{},字典类别：{}'.format(self.classes, self.dict_classes))

        # 获得训练数据集加载器
        data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle)
        return data_loader



