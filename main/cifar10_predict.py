import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50, resnet34
from config.cifar10_config import Cifar10Config
from config.test_config import TestConfig
from config.mnist_config import MnistConfig
from config.catdog_config import CatDogConfig
from train_and_test.train_and_valid import train_and_valid, train_and_valid_, test
from models.alexnet import AlexNet
from models import resnet_v2, vggnet
from utils.tools import visiual_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import cv2
from PIL import Image
from models.lenet import LeNet

# ----------------配置数据--------------------------
# 配置实例化
# cfg = Cifar10Config()
cfg = MnistConfig()
# 数据预处理
test_data_preprocess = transforms.Compose([
                                           transforms.ToTensor()]
                                           )

# 获取测试集的加载器
test_loader = cfg.dataset_loader(root=cfg.mnist_dir, train=False, shuffle=False,
                                 data_preprocess=test_data_preprocess)
# ---------------构建网络、定义损失函数、优化器--------------------------
# net = resnet18()
# net = resnet_v2.resnet18(num_classes=cfg.num_classes, type_dataset='cifar-10')
# net = vggnet.VGG(vgg_name='VGG11', num_classes=10, dataset='cifar-10')
net = LeNet(num_classes=10)
# 重写网络最后一层
# fc_in_features = net.fc.in_features  # 网络最后一层的输入通道
# net.fc = nn.Linear(in_features=fc_in_features, out_features=cfg.num_classes)
# 加载模型权重、将网络放入GPU
if os.path.exists(cfg.checkpoints):
    net.load_state_dict(torch.load(cfg.checkpoints))
    print('load model argument...')

# -------------进行测试-----------------
print('进行测试.....')
# 测试函数
print('test stage...\n')


# 将网络结构调成验证模式、定义准确率、标签列表和预测列表
net.eval()
# cat_file = '/home/team/xiaonan/Dataset/cat_dog/test/cat/'
# dog_file = '/home/team/xiaonan/Dataset/cat_dog/test/'
#
# cat_list = os.listdir(cat_file)
# dog_list = os.listdir(dog_file)

# with torch.no_grad():
#     for index, image in enumerate(cat_list):
#         image = os.path.join(cat_file, image)
#         image = Image.open(image)
#         image = image.resize(size=(224, 224))
#         input = transforms.ToTensor()(image)
#         input = torch.unsqueeze(input, 0)
#         print(input.size())
#         outputs = net(input)
#         outputs = nn.functional.softmax(outputs, dim=1)
#         pred = torch.argmax(outputs, dim=1)
#
#         image = input.squeeze(0).numpy().transpose(1, 2, 0)
#         info = 'True label:{}, Predict:{}'.format(cfg.name_classes[3],
#                                                   cfg.name_classes[pred])
#         plt.title(info)
#         plt.imshow(image)
#         plt.savefig( 'cat_'+str(index)+ '.png')
#         plt.show()
#         plt.close()
#         print(input.size())

# 进行测试集的测试
with torch.no_grad():  # 不使用梯度，减少内存占用
    for images, labels in test_loader:
        # 将测试数据放入GPU上
        # images, labels = images.to(cfg.device), labels.to(cfg.device)

        # 推理输出网络预测值，并使用softmax使预测值满足0-1概率范围
        outputs = net(images)
        outputs = nn.functional.softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1)
        for i in range(labels.size(0)):
            if labels[i] == 8:
                image = images[i].cpu().numpy()
                image = image.transpose(1, 2, 0)
                image = np.squeeze(image, 2)
                print(image.shape)
                # image = np.resize(image, (32, 32))
                # image = cv2.resize(image, (32, 32))
                print(image.shape)

                print(image.shape)
                info = 'True label:{}, Predict:{}'.format(cfg.name_classes[labels[i]],
                                                          cfg.name_classes[pred[i]])
                plt.title(info)
                plt.imshow(image)
                plt.savefig(str(i) + '.png')
                plt.show()
                plt.close()

        # break


        # # 计算每个预测值概率最大的索引（下标）；统计真实标签和对应预测标签
        # correct += torch.sum((pred == labels)).item()
        # targets += list(labels.cpu().numpy())
        # preds += list(pred.cpu().numpy())

# 计算测试精度和混淆矩阵
# test_acc = 100. * correct / len(test_loader.dataset)
# confusion_mat = metrics.confusion_matrix(targets, preds)
# confusion_mat = confusion_matrix(targets, preds)
# print('numbers samples:{}, test accuracy:{},\nconfusion matrix:\n{}'.
#      format(len(test_loader.dataset), test_acc, ''))

