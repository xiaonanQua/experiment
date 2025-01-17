import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from config.cifar100_config import Cifar100Config
from train_and_test.train_and_valid import train_and_valid_, test
from models.backbone import resnet_v2
from utils.tools import visiual_confusion_matrix
from utils.warmup_optim import WarmupOptimizer


# ----------------配置数据--------------------------
# 配置实例化
cfg = Cifar100Config()

# 获取数据集的均值和标准差
cfg.get_mean_and_std(root=cfg.cifar_100_dir)

# 均值和标准差
mean = [0.49139961, 0.48215843, 0.44653216]
std = [0.24703216, 0.2434851, 0.26158745]

# 数据预处理
train_data_preprocess = transforms.Compose([transforms.Resize(size=(224, 224)),
                                            # transforms.RandomResizedCrop(224),
                                            # transforms.RandomHorizontalFlip(),
                                            # transforms.ColorJitter(brightness=0.4, saturation=0.4,
                                            #                        hue=0.4, contrast=0.4),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean,
                                                                 std=std)])
valid_data_preprocess = transforms.Compose([transforms.Resize(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=mean,
                                                                std=std)])

test_data_preprocess = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=mean,
                                                                std=std)])

# 获取训练集、测试集的加载器
# train_loader, valid_loader = cfg.dataset_loader(root=cfg.cifar_10_dir, train=True,
#                                                 data_preprocess=[train_data_preprocess, valid_data_preprocess],
#                                                 valid_coef=0.1)

train_loader = cfg.dataset_loader(root=cfg.cifar_100_dir, train=True,
                                 data_preprocess=train_data_preprocess)
test_loader = cfg.dataset_loader(root=cfg.cifar_100_dir, train=False, shuffle=False,
                                 data_preprocess=valid_data_preprocess)

# ---------------构建网络、定义损失函数、优化器--------------------------
# 构建网络结构
# net = resnet(num_classes=cfg.num_classes)
# net = AlexNet(num_classes=cfg.num_classes)
# net = resnet50(num_classes=cfg.num_classes)
# net = resnet18(num_classes=cfg.num_classes)
net = resnet_v2.resnet18(num_classes=cfg.num_classes)
# net = vggnet.VGG(vgg_name='VGG11', num_classes=10, data2='cifar-10')

# 将网络结构、损失函数放置在GPU上；配置优化器
net = net.to(cfg.device)
# net = nn.DataParallel(net, device_ids=[0, 1])
criterion = nn.CrossEntropyLoss().cuda(device=cfg.device)
# 常规优化器：随机梯度下降和Adam
optimizer = optim.SGD(params=filter(lambda p:p.requires_grad, net.parameters()), lr=0,
                      weight_decay=cfg.weight_decay, momentum=cfg.momentum)
optimizer = WarmupOptimizer(lr_base=cfg.learning_rate, optimizer=optimizer, data_size=cfg.data_size,
                            batch_size=cfg.batch_size, is_warmup=False)
# optimizer = optim.Adam(params=net.parameters(), lr=cfg.learning_rate,
#                        weight_decay=cfg.weight_decay)
# 线性学习率优化器
#optimizer = optim.SGD(params=net.parameters(), lr=cfg.learning,
                     # weight_decay=cfg.weight_decay, momentum=cfg.momentum)

# --------------进行训练-----------------
print('进行训练....')
train_and_valid_(net, criterion=criterion,
                 optimizer=optimizer,
                 train_loader=train_loader,
                 valid_loader=test_loader, cfg=cfg)

# -------------进行测试-----------------
print('进行测试.....')
test_accs, confusion_mat = test(net, test_loader, cfg)

graph_name = cfg.model_name + ' Accuracy:'+str(test_accs)
# -------------可视化-------------------
visiual_confusion_matrix(confusion_mat, cfg.name_classes, graph_name=graph_name, out_path=cfg.result_dir)

