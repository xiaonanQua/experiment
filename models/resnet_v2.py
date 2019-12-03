"""
实现了18,34,50,101,152层的resnet网络结构，并对于大图像的数据集（例如ImageNet）或者小图像的数据集(例如cifar-10)
第一层的卷积核进行相应的调整来适应不同图像大小的数据集
"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from models.network_util import BasicBlock, BottleBlock

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pt',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}


class ResNet(nn.Module):
    """
    残差网络
    """
    def __init__(self, num_classes, type_block, layers, type_dataset=None):
        """
        定义残差网络的组件
        :param num_classes: 类别数量
        :param type_block: 残差块对象，分为基础块和瓶颈块
        :param layers: 残差块的数量
        :param type_dataset:数据集类型，对于类似于cifar10这种图像尺寸比较小的数据集，应修改第一步的卷积核。
        """
        super(ResNet, self).__init__()
        self.inplanes = 64  # 通道数
        # 定义网络组件
        if type_dataset in ['cifar-10']:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(type_block, 64, layers[0])
        self.layer2 = self._make_layer(type_block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(type_block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(type_block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*type_block.expansion, num_classes)

    def forward(self, x):
        # 第一组对输入图像进行一个卷积
        x = self.conv1(x)
        print(x.size())
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        # 四个组
        x = self.layer1(x)
        print(x.size())
        x = self.layer2(x)
        print(x.size())
        x = self.layer3(x)
        print(x.size())
        x = self.layer4(x)
        print(x.size())

        # 平均池化再添加一个全连接
        x = self.avgpool(x)
        print(x.size())
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, planes, num_block, stride=1):
        """
        使用基础块或者瓶颈块来制作多组相同的层
        :param block: 块类型对象，基础块或者瓶颈块对象
        :param planes: 输入层的通道数
        :param num_block: 块的数量，list类型
        :param stride: 步长
        :return:
        """
        downsaple = None
        # 定义下采样操作
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsaple = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsaple))
        self.inplanes = planes*block.expansion
        for i in range(1, num_block):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


def resnet18(num_classes, pretrained=False, **kwargs):
    """
    构建一个ResNet18的模型
    :param num_classes:类别数量
    :param pretrained:如果为True，返回一个在ImageNet上预训练的模型
    :param kwargs:函数中的命名参数
    :return:
    """
    model = ResNet(num_classes=num_classes, type_block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'),
                              strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """
    构建一个ResNet34的模型
    :param num_classes:类别数量
    :param pretrained:如果为True，返回一个在ImageNet上预训练的模型
    :param kwargs:函数中的命名参数
    :return:
    """
    model = ResNet(num_classes=num_classes, type_block=BasicBlock, layers=[2, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'),
                              strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """
    构建一个ResNet34的模型
    :param num_classes:类别数量
    :param pretrained:如果为True，返回一个在ImageNet上预训练的模型
    :param kwargs:函数中的命名参数
    :return:
    """
    model = ResNet(num_classes=num_classes, type_block=BottleBlock, layers=[3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'),
                              strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """
    构建一个ResNet34的模型
    :param num_classes:类别数量
    :param pretrained:如果为True，返回一个在ImageNet上预训练的模型
    :param kwargs:函数中的命名参数
    :return:
    """
    model = ResNet(num_classes=num_classes, type_block=BottleBlock, layers=[3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'),
                              strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """
    构建一个ResNet34的模型
    :param num_classes:类别数量
    :param pretrained:如果为True，返回一个在ImageNet上预训练的模型
    :param kwargs:函数中的命名参数
    :return:
    """
    model = ResNet(num_classes=num_classes, type_block=BottleBlock, layers=[3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'),
                              strict=False)
    return model


if __name__ == '__main__':
    # a = torch.randn(2)
    # print(a)
    # print(nn.ReLU()(a))
    # print(nn.ReLU(inplace=True)(a))
    a = torch.randn(2, 3, 32, 32)
    net = resnet152(10, type_dataset = 'cifar-10')
    print(net)
    print(net(a))
