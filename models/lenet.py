import torch
import torch.nn as nn
import torch.nn.functional as func


class LeNet(nn.Module):
    def __init__(self, num_classes):
        # 继承父类构造函数
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        # 卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        feature = self.conv(x)
        output = self.fc(feature.view(x.size(0), -1))
        return output


if __name__ == '__main__':
    net = LeNet()
    print(net)
    # 返回网络中的学习参数
    params = list(net.parameters())
    print(params)
    # 输出网络中的学习参数及其名称
    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())
    print()

    # forward函数的输入和输出都是Variable，只有Variable才具有自动求导的功能，Tensor是没有的，所以在输入时需要把Tensor封装成Variable
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out.size())
    # 所有参数的梯度清零
    net.zero_grad()
    out.backward(torch.randn(1, 10))
    target = torch.randn(10)
    target = target.view(1, -1)
    print(target)
    criterion = nn.MSELoss()
    loss = criterion(input=input, target=target)
    print(loss)
