import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from config.catdog_config import CatDogConfig
from train_and_test.train_and_valid import test
from models.backbone.alexnet import AlexNet

# ----------------配置数据--------------------------
# 配置实例化
cfg = CatDogConfig()
# cfg = TestConfig()

mean = [0.49139961, 0.48215843, 0.44653216]
std = [0.24703216, 0.2434851, 0.26158745]

# 数据预处理
train_data_preprocess = transforms.Compose([transforms.Resize(size=(224,224)),
                                            #transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(brightness=0.4, saturation=0.4,
                                                                   hue=0.4, contrast=0.4),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=cfg.mean,
                                                                 std=cfg.std)])
valid_data_preprocess = transforms.Compose([transforms.Resize(size=(224, 224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=cfg.mean,
                                                                std=cfg.std)])

# test_data_preprocess = transforms.Compose([transforms.Resize(256),
#                                            transforms.CenterCrop(224),
#                                            transforms.ToTensor(),
#                                            transforms.Normalize(mean=cfg.mean,
#                                                                 std=cfg.std)])

# 获取训练集、测试集的加载器
# train_loader, valid_loader = cfg.dataset_loader(root=cfg.cat_dog_train, train=True,
#                                                 data_preprocess=[train_data_preprocess, valid_data_preprocess],
#                                                 valid_coef=0.1)

train_loader = cfg.dataset_loader(root=cfg.cat_dog_train, train=True,
                                  data_preprocess=train_data_preprocess)
valid_loader = cfg.dataset_loader(root=cfg.cat_dog_valid, train=True,
                                  data_preprocess=valid_data_preprocess)
# test_loader = cfg.dataset_loader(root=cfg.cat_dog_test, train=False, shuffle=False,
#                                  data_preprocess=valid_data_preprocess)

# ---------------构建网络、定义损失函数、优化器--------------------------
# 构建网络结构
# net = resnet()
net = AlexNet(num_classes=cfg.num_classes)
# net = resnet50()
#net = resnet18()
# 重写网络最后一层
#fc_in_features = net.fc.in_features  # 网络最后一层的输入通道
#net.fc = nn.Linear(in_features=fc_in_features, out_features=cfg.num_classes)

# 将网络结构、损失函数放置在GPU上；配置优化器
net = net.to(cfg.device)
# net = nn.DataParallel(net, device_ids=[0, 1])
# criterion=nn.BCELoss()
#criterion = nn.BCEWithLogitsLoss().cuda(device=cfg.device)
criterion = nn.CrossEntropyLoss().cuda(device=cfg.device)
# 常规优化器：随机梯度下降和Adam
#optimizer = optim.SGD(params=net.parameters(), lr=cfg.learning_rate,
#                      weight_decay=cfg.weight_decay, momentum=cfg.momentum)
optimizer = optim.Adam(params=net.parameters(), lr=cfg.learning_rate,
                       weight_decay=cfg.weight_decay)
# 线性学习率优化器
#optimizer = optim.SGD(params=net.parameters(), lr=cfg.learning,
                     # weight_decay=cfg.weight_decay, momentum=cfg.momentum)

# --------------进行训练-----------------
# print('进行训练....')
# train_and_valid_(net, criterion=criterion,
#                  optimizer=optimizer,
#                  train_loader=train_loader,
#                  valid_loader=valid_loader, cfg=cfg,
#                  is_lr_warmup=False, is_lr_adjust=False)

# -------------进行测试-----------------
print('进行测试.....')
test_accs, confusion_mat = test(net, valid_loader, cfg)

# -------------可视化-------------------
# visiual_confusion_matrix(confusion_mat, cfg.name_classes, graph_name=cfg.model_name, out_path=cfg.result_dir)
