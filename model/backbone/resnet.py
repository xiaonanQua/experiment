import torch
import torch.nn as nn
import torchvision


def conv3x3(in_channels, out_channels, stride=1):
    """
    定义通用卷积核为3X3的卷积
    :param in_channels: 输入通道
    :param out_channels: 输出通道
    :param stride: 步长
    :return:
    """
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class ResidualBlock(nn.Module):
    """
    构建残差模块（浅层网络），残差单元解决了深度网络中梯度消失的问题，导致训练效果很差。也就是随着网络的层次，出现退化问题（准确率下降）。
    这不是过拟合的结果，若是过拟合的话，应该准确率会很高。使用残差单元构建深层网络，梯度也不会消失。相较于原始单元更易收敛，
    有一定的正则化效果。
    网络结构（浅层网络）：
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 若输入通道与输出通道不一致，则对输入进行下采样

    def forward(self, x):
        identity = x  # 恒等映射
        # 进行残差计算
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        if self.downsample:  # 若下采样存在，则对输入进行下采样使得输出和输入的通道一致
            identity = self.downsample(x)
        # 得到特征
        out = residual + identity
        out = self.relu(out)  # 对网络进行激活
        return out


class ResNet(nn.Module):
    """
    使用残差单元进行残差网络的构建
    """
    def __init__(self, block, layers, num_classes=10):
        """
        构造函数
        :param block: 残差学习单元（对象）
        :param layers: 构造层（函数），构造特定的层
        :param num_classes: 输出的类别数量
        """
        super(ResNet, self).__init__()
        self.in_channels = 16  # 输入通道
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, num_block, stride=1):
        """

        :param block: 残差模块
        :param out_channels: 输出通道
        :param num_block: 残差块的数量
        :param stride: 步长
        :return:
        """
        downsample = None  # 下采样
        # 步长不为1，输入通道不等于输出通道
        if (stride !=1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(conv3x3(self.in_channels, out_channels, stride=stride),
                                       nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, num_block):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out




