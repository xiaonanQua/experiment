import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as model
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from config.resnet_cifar10_config import ResNetConf
from models.resnet import ResidualBlock
import train_and_test.train_and_valid as train
import train_and_test.test as test


# --------------------------------------------配置文件

# 获取ResNet50的网络结构
net = model.resnet50(pretrained=True, progress=True)
# 重写网络的最后一层
fc_in_features = net.fc.in_features
net.fc = nn.Linear(fc_in_features, 10)

# 显示网络结构参数
for name, child in net.named_children():
    for name2, params in child.named_parameters():
        print(name,name2)

# 实例化配置文件
cfg = ResNetConf()

# 若设备支持gpu加速，则将模式放置gpu中进行训练
if cfg.use_gpu:
    net.cuda()

# -------------------------------------------读取数据
# 对图像进行一系列预处理
data_preprocess = transforms.Compose([
    transforms.Resize(40),  # 对图像的大小进行重塑
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(32),
    transforms.ToTensor()
    ])

# 获取数据集,包括训练集，验证集，测试集
train_dataset = CIFAR10(root=cfg.cifar_10_dir, train=True, transform=data_preprocess, download=True)
# val_dataset = CatDog(root=cfg.catdog_train_dir, low_memory=False)
test_dataset = CIFAR10(root=cfg.cifar_10_dir, train=False, transform=transforms.ToTensor)

# 获取数据集（低内存版）
# train_dataset = CatDog(root=cfg.catdog_train_dir, train=True)
# val_dataset = CatDog(root=cfg.catdog_train_dir)
# test_dataset = CatDog(root=cfg.catdog_test_dir, test=True, low_memory=False)

# 通过数据加载器加载数据
train_data_loader = DataLoader(train_dataset, cfg.batch_size,
                               shuffle=True, num_workers=cfg.num_workers,)
# val_data_loader = DataLoader(val_dataset, batch_size=32,
                               # shuffle=True, num_workers=cfg.num_workers)
test_data_loader = DataLoader(test_dataset, cfg.batch_size, num_workers=cfg.num_workers)

# 通过数据加载器加载数据（低内存版本）
# train_data_loader = DataLoader(train_dataset, num_workers=cfg.num_workers)
# val_data_loader = DataLoader(val_dataset, num_workers=cfg.num_workers)
# test_data_loader = DataLoader(test_dataset, num_workers=cfg.num_workers)

# -------------------------------------------------------------定义训练器、优化器并进行训练和验证

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=cfg.learning_rate,
                             weight_decay=cfg.weight_decay)

# 进行训练,返回模型保存路径
model_path = train.train(model=net, train_data_loader=train_data_loader,
                         criterion=criterion, optimizer=optimizer, cfg=cfg)
# 实例化model对象
# model = getattr(models, cfg.model)(ResidualBlock, [3, 3, 3])
# # 测试
# # model_path = '../checkpoints/AlexNet_0913_15:41:50.pth'
# print('model save path:{}'.format(model_path))
# # 进行测试
# test.test(model=model, model_path=model_path, test_data_loader=test_data_loader,
#           train_data_loader=train_data_loader)












