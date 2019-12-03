from torch.nn import init
import torch
import torch.nn as nn
import numpy as np


def init_weight(net, zero_gamma=False, init_type='normal', gain=0.02):
    def init_func(m):
        # print('m:', m)
        classname = m.__class__.__name__
        # print('class name',classname)
        # print(classname.find)
        if zero_gamma:
            if hasattr(m, 'bn2'):
                init.constant_(m.bn2.weight.data, 0.0)
                init.constant_(m.bn2.bias.data, 0.0)
                print(1)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)
                print(2)
        elif classname.find('BatchNorm2d') != -1:
            if zero_gamma:
                init.constant_(m.weight.data, 0.0)
                init.constant_(m.bias.data, 0.0)
                print(3)
            else:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)
                print(4)

    net.apply(init_func)


def show_network_param(net,  data_and_grad=False):
    """
    显示网络中的参数
    :param net: 网络结构
    :param data_and_grad: 是否显示数据和梯度
    :return:
    """
    print('print network parameters...\n')
    for name, param in net.named_parameters():
        print(name, param.size())
        if data_and_grad is True:
            print(name, param.data.shape, param.grad)


def parameter_initial(net):
    """
    对网络结构的参数进行初始化
    :param net: 需要进行参数初始化的网络
    :return:
    """
    for name, param in net.named_parameters():
        if 'weight' in name:
            init.normal_(param, mean=0, std=0.01)
            print(name, param.data)
        if 'bias' in name:
            init.constant_(param, val=0.0)
    return net

def conv2d(x, kernel):
    """
    二维卷积运算（互相关）
    :param x: 二维数据
    :param kernel: 二维卷积核
    :return: 经过卷积运算后的数据
    """
    # 获得卷积核的高度和宽度
    height, width = kernel.size()
    # 初始化经过卷积后的二维数组
    y = torch.zeros(size=(x.size(0) - height + 1, x.size(1) - width + 1))
    # print('卷积后的形状：{}'.format(y.size()))
    for i in range(y.size(0)):
        for j in range(y.size(1)):
            # 进行卷积运算，并更新
            y[i, j] = (x[i:height+i, j:width+j]*kernel).sum()
    return y


class Conv2D(nn.Module):
    """
    自定义二维卷积层，包括权重和偏差参数
    """
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))
        print(self.weight, self.bias)

    def forward(self, x):
        return conv2d(x, self.weight) + self.bias


def simple_example():
    """
    简单实现卷积的前向传播和反向传播
    :return:
    """
    x = torch.tensor([[1, 1, 1, 1, 1], [-1, 0, -3, 0, 1],
                      [2, 1, 1, -1, 0], [0, -1, 1, 2, 1],
                      [1, 2, 1, 1, 1]])
    conv = Conv2D(kernel_size=(3, 3))

    step = 50
    lr = 0.01
    y = torch.ones(3, 3)
    y[:, 1:3] = 0
    print(y)

    for i in range(step):
        y_pred = conv(x.float())
        loss = ((y - y_pred)**2).sum()
        loss.backward()

        # 梯度下降
        conv.weight.data = conv.weight.data - lr*conv.weight.grad
        conv.bias.data = conv.bias.data - lr*conv.bias.grad

        # 梯度清0
        conv.weight.grad.fill_(0)
        conv.bias.grad.fill_(0)
        print('{},{}'.format(i, loss))


class BasicBlock(nn.Module):
    expansion = 1
    """
    ResNet的基础块，包含两层卷积核为3*3的卷积的残差块和恒等映射
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 恒等映射
        identity = x

        # 构建两块残差映射
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 若下采样不为空，则进行下采样，以使残差映射和恒等映射能够相加
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差映射和恒等映射能够相加
        out += identity
        out = self.relu(out)
        print(out.size(), identity.size())

        return out


class BottleBlock(nn.Module):
    expansion = 4
    """
    ResNet的瓶颈块，用于构建更深的网络（也是对于像ImageNet这样的数据集）。
    使用三层的堆叠，分别是1*1,3*3,1*1卷积。其中1×1层负责减小然后增加（还原）尺寸，
    而3×3层则成为输入/输出尺寸较小的瓶颈。
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 恒等映射
        identity = x

        # 残差映射
        # 1×1卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3*3卷积
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1*1卷积
        out = self.conv3(out)
        out = self.bn3(out)

        # 判断是否进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差映射和恒等映射相加
        out += identity
        out = self.relu(out)
        print(out.size())

        return out


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        x = x.view(x.shape[0], -1)


def block_vgg(num_convs, in_channels, out_channels):
    block = []
    # 使用几层３＊３卷积来代替大卷积核的卷积，在保证相同的感受野的情况下，提升网络的深度，提高网络效果
    for i in range(num_convs):
        if i == 0:
            block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1))
        else:
            block.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1))
    # 添加最大池化层，使宽、高减半，进行降维
    block.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*block)



if __name__ == '__main__':
    data = torch.randn(size=(10, 64, 32, 32))
    data2 = torch.randn(size=(10, 3, 32, 32))
    a = np.ones(shape=[2, 2])
    b = np.ones(shape=[2,2])
    print(a+b)
    # print(data)
    basic_block = BasicBlock(64, 64)

    # show_network_param(basic_block, data_and_grad=True)
    out = basic_block.forward(data)
    # print(out.shape)
    # print(basic_block)






