import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
from config.cifar10_config import Cifar10Config
from config.test_config import TestConfig
from train_and_test.train_and_valid import train_and_valid_gpus, test
from models.alexnet import AlexNet
from utils.tools import vis


# ----------------配置数据--------------------------
# 配置实例化
cfg = Cifar10Config()
# cfg = TestConfig()

mean = [0.49139961, 0.48215843, 0.44653216]
std = [0.24703216, 0.2434851, 0.26158745]

# 数据预处理
train_data_preprocess = transforms.Compose([transforms.Resize(size=(224, 224)),
                                            # transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(brightness=0.4, saturation=0.4,
                                                                   hue=0.4, contrast=0.4),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=cfg.mean,
                                                                 std=cfg.std)])
valid_data_preprocess = transforms.Compose([transforms.Resize(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=cfg.mean,
                                                                std=cfg.std)])

test_data_preprocess = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=cfg.mean,
                                                                std=cfg.std)])

# 获取训练集、测试集的加载器
# train_loader, valid_loader = cfg.dataset_loader(root=cfg.cifar_10_dir, train=True,
#                                                 data_preprocess=[train_data_preprocess, valid_data_preprocess],
#                                                 valid_coef=0.1)

train_loader = cfg.dataset_loader(root=cfg.cifar_10_dir, train=True,
                                  data_preprocess=train_data_preprocess)
test_loader = cfg.dataset_loader(root=cfg.cifar_10_dir, train=False, shuffle=False,
                                 data_preprocess=valid_data_preprocess)

# ---------------构建网络、定义损失函数、优化器--------------------------
# 构建网络结构
# net = resnet()
# net = AlexNet(num_classes=cfg.num_classes)
net = resnet50()
# 重写网络最后一层
fc_in_features = net.fc.in_features  # 网络最后一层的输入通道
net.fc = nn.Linear(in_features=fc_in_features, out_features=cfg.num_classes)

# 将网络结构、损失函数放置在GPU上；配置优化器
# net.to(cfg.device)
net = nn.DataParallel(net, device_ids=[0,1])
criterion = nn.CrossEntropyLoss().cuda()
# 常规优化器：随机梯度下降和Adam
optimizer = optim.SGD(params=net.parameters(), lr=cfg.learning_rate,
                     weight_decay=cfg.weight_decay, momentum=cfg.momentum)
# optimizer = optim.Adam(params=net.parameters(), lr=cfg.learning_rate,
#                        weight_decay=cfg.weight_decay)
# 线性学习率优化器
#optimizer = optim.SGD(params=net.parameters(), lr=cfg.learning,
                     # weight_decay=cfg.weight_decay, momentum=cfg.momentum)

# --------------进行训练-----------------
print('进行训练....')
train_and_valid_gpus(net, criterion=criterion,
                 optimizer=optimizer,
                 train_loader=train_loader,
                 valid_loader=test_loader, cfg=cfg,
                 is_lr_warmup=False, is_lr_adjust=True)

# -------------进行测试-----------------
print('进行测试.....')
test_accs, confusion_mat = test(net, test_loader, cfg)

# -------------可视化-------------------
# print(test_accs, confusion_mat)
# vis(test_accs, confusion_mat, cfg.classes)

