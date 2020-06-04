import os
from torch.utils import data
from config.config import Config
from torchvision import transforms as tf
from PIL import Image


class CatDog(data.Dataset):
    def __init__(self, root, transforms=None, train=False, test=False, low_memory=True):
        """
        继承DataSet父类，获取所有图片地址，并根据训练、验证、测试划分数据
        :param root: 数据根目录
        :param transforms: 转化数据的一系列函数
        :param train: 训练
        :param test: 测试
        :param low_memory: boolean类型，若设备内存过于低，无法处理大内存数据，则只测试小数据用于程序调试。
        """
        super(CatDog, self).__init__()
        self.test = test
        self.low_memory = low_memory
        self.class_name = {0: 'cat', 1: 'dog'}
        # 将数据目录和数据名称结合在一起，保存到列表中
        images_path_list = [os.path.join(root, image_name) for image_name in os.listdir(root)]
        if self.test:
            # 将图像路径列表进行排序，根据文件下标
            images_path_list = sorted(images_path_list, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            # 将图像路径列表进行排序，根据文件下标
            images_path_list = sorted(images_path_list, key=lambda x: int(x.split('.')[-2]))

        num_images = len(images_path_list)  # 图像文件数量

        # 划分训练、验证集，验证：训练=3:7
        if self.test:
            self.images = images_path_list
        elif train:
            self.images = images_path_list[:int(0.7*num_images)]
        else:
            self.images = images_path_list[int(0.7*num_images):]

        # 设置transforms
        if transforms is None:
            # 数据归一化
            normalize = tf.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            # 测试集和验证集的数据转化操作(对数据进行预处理)
            if self.test or not train:
                self.data_preprocess = tf.Compose([
                    tf.Resize(224),  # 重塑图像大小
                    tf.CenterCrop(224),
                    tf.ToTensor(),
                    normalize
                ])
            else:
                # 训练集的转换
                self.data_preprocess = tf.Compose([
                    tf.Resize(256),
                    tf.RandomResizedCrop(224),  # 将给的图片裁剪出随机大小和纵横比
                    tf.RandomHorizontalFlip(),  # 按照一定概率随机翻转水平图片
                    tf.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """
        将文件读取等费时操作放在__getitem__函数中，利用多进程加速。
        :param index: 图片id
        :return:
        """
        # 根据id获得一张图片的路径
        image_path = self.images[index]
        # 通过图像数据名称划分出标签名称
        if self.test:
            label = int(image_path.split('.')[-2].split('/')[-1])
            print(label)
        else:
            label = 1 if 'dog' in image_path.split('/')[-1] else 0
        # 读取图片数据
        data = Image.open(image_path)
        # 预处理数据
        data = self.data_preprocess(data)

        return data, label

    def __len__(self):
        """
        返回数据集中所有图片的个数,
        :return:
        """
        # 若设备是低内存，则只返回长度为100的数据
        if self.low_memory:
            return 100
        else:  # 所以返回所有图片数量
            return len(self.images)


if __name__ == '__main__':
    cfg = Config()
    train_data = CatDog(root=cfg.catdog_train_dir, train=True)
    print(len(train_data))
    images, label = iter(train_data[1])
    print(images, label)
    valid_data = CatDog(root=cfg.catdog_train_dir)
    print(len(valid_data))
    test_data = CatDog(root=cfg.catdog_test_dir, test=True)
    print(len(test_data))

