"""
实现VGGNet网络结构
"""
import torch
import torch.nn as nn
import models.network_util as net_tool

model_struct = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGGNet(nn.Module):

    def __init__(self, conv_list, num_classes):
        super(VGGNet, self).__init__()
        self.conv_list = ((1, 3, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
        # 经过５个vgg block，宽高会减半５次，所以224/32 = 7
        self.fc_features = 512*7*7
        self.fc_hidden_units = 4096

        self.fc_start = nn.Linear(self.fc_features, self.fc_hidden_units)
        self.fc_middle = nn.Linear(4096, 4096)
        self.fc_out = nn.Linear(4096, num_classes)

    def forward(self, x):
        for i, (num_conv, in_planes, out_planes) in enumerate(self.conv_list):
            x = net_tool.block_vgg(num_conv, in_planes, out_planes)(x)


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, dataset):
        super(VGG, self).__init__()
        self.features = self._make_layers(model_struct[vgg_name])
        if dataset == 'cifar-10':
            self.classifier = nn.Linear(512, num_classes)
        else:
            self.classifier = nn.Sequential(nn.Linear(512*7*7, 4096),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(4096, 4096),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(4096, num_classes))

    def _make_layers(self, model_struct):
        layers = []
        in_channels = 1
        for x in model_struct:
            if x is 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    x = torch.randn(size=(3, 3, 32, 32))
    net = VGG('VGG11', 10, 'cifar-10')
    print(net(x).shape)

