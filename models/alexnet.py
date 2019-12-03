import torch.nn as nn
import torch.nn.functional as F
import torch
from models import network_util


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.network_name = 'AlexNet'  # 模型名称
        self.image_height = 227
        self.image_width = 227

        # 定义特征序列
        self.features = nn.Sequential(
            # 卷积层1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),  # (227->55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 55->27
            # 卷积层2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # 27->27
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27->13
            # 卷积层3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # 13->13
            nn.ReLU(),
            # 卷积层4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # 13->13
            nn.ReLU(),
            # 卷积层5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 13->13
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 13->6
        )

        # 定义全连接序列
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self,x):
        """
        前向传播
        :return:
        """
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # 将特征的结果形状相乘
        logit = self.classifier(x)
        return logit


if __name__ == '__main__':
    images = torch.randn(10, 3, 224, 224)
    net = AlexNet()
    print(net)
    # outputs = net(images)
    # outputs = F.softmax(outputs, dim=1)
    # print(outputs)
    network_util.show_network_param(net, data_and_grad=True)
    network_util.parameter_initial(net)
    network_util.show_network_param(net, data_and_grad=True)

