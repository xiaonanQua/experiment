#  -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision.models as model
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
# import ResNet as resnet
# import res2net as res2net
# import res2next as res2next
# import dla as dla
# from sampler import ImbalancedDatasetSampler
import time
import copy
import os

# 训练、测试数据集路径
train_dataset_path = '/home/data/V1.0/train/'
test_dataset_path = '/home/data/V1.0/test/'

# 设置实验超参数
num_classes = 4
num_epoch = 200
batch_size = 32
learning_rate = 0.0001
weight_decay = 0.0001
momentum = 0.9
keep_prob = 0.5

# 类别名称和设备
class_name = None  # 类别名称
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 平均值和标准差
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 保存模型路径
# model_path = '/root/notebook/model/river_res2net50.pth'
model_path = '/root/notebook/model/river_res2net101.pth'
# model_path = '/root/notebook/model/river_res2next50.pth'
# model_path = '/root/notebook/model/river_res2nextdla.pth'
# model_path = '/root/notebook/model/river_res2netdla.pth'
# model_path = '/root/notebook/model/river_vgg_bn_19.pth'

# 对数据进行预处理
data_preprocess = transforms.Compose([
    #     transforms.Resize(size=(256, 256)),  # 将输入PIL图像的大小调整为给定大小。
    # 随机裁剪出一块面积为原面积的10%区域,然后再将区域的宽和高缩放到112像素，随机概率在[0.5,2]中去个值
    transforms.RandomResizedCrop(224, scale=(0.1, 1), ratio=(0.5, 2)),
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平翻转给定的PIL图像。
    # transforms.RandomVerticalFlip(),  # 随机垂直(上下)翻转
    # 改变图像的颜色，随机变化图像的亮度，对比度，饱和度和色调
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.ToTensor(),  # 将PIL格式的图像转化成tensor对象，值在0-1之间
    transforms.Normalize(mean=mean, std=std)  # 将图像数据进行标准归一化处理
])

# 加载数据集
image_datasets = ImageFolder(root=train_dataset_path, transform=data_preprocess)
class_name = image_datasets.classes
# print(image_datasets.imgs[:30])
print(image_datasets.class_to_idx)
print(image_datasets[0][0].size(), image_datasets[1][0].size())
print(class_name)

# 数据加载器;进行重采样后的数据加载器
train_data_loader = DataLoader(dataset=image_datasets, batch_size=batch_size, shuffle=True)
# train_data_loader = DataLoader(dataset=image_datasets, batch_size=batch_size,
#                                sampler=ImbalancedDatasetSampler(image_datasets))

# 定义模型
# 获取ResNet50的网络结构
net = model.resnet50(pretrained=True, progress=True)
# net = res2net.res2net50_26w_8s(pretrained=True)
# net = res2net.res2net101_26w_4s(pretrained=True)
# net = res2next.res2next50(pretrained=True)
# net = dla.res2next_dla60(pretrained=True)
# net = dla.res2net_dla60(pretrained=True)
# net = model.vgg19_bn(pretrained=True, progress=True)
# print(net.classifier[6])


# 重写网络的最后一层
fc_in_features = net.fc.in_features
net.fc = nn.Linear(fc_in_features, num_classes)
# net.classifier[6] = nn.Linear(4096, num_classes)
# print(net.fc)
# print(net)

# dla网络结构，最后一层的重写
# fc = nn.Conv2d(1024, 4, kernel_size=1, stride=1, padding=0, bias=True)
# net.fc = fc
# print(net.fc)

# 若模型已经训练存在，则加载参数继续进行训练
if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path))
    print('加载检查点...')
# 将网络结构放置在gpu上
net.to(device)

# 显示网络结构参数s
# for name, child in net.named_children():
#     for name2, params in child.named_parameters():
#         print(name,name2)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()

outputs_params = list(map(id, net.fc.parameters()))  # 输出层参数
feature_params = filter(lambda p: id(p) not in outputs_params, net.parameters())  # 特征参数
# optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer = torch.optim.SGD(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
# optimizer = torch.optim.SGD([{'params': feature_params},
#                              {'params': net.classifier[6].parameters(), 'lr':learning_rate*10}],
#                             lr=learning_rate, weight_decay=weight_decay)
# optimizer = torch.optim.Adam([{'params': feature_params},
#                              {'params':net.fc.parameters(), 'lr':learning_rate*10}],
#                              lr=learning_rate, weight_decay=weight_decay)

# ----------------------------进行训练------------------

# 训练的开始时间
start_time = time.time()

# 深层复制模型的状态字典（模型的参数）， 定义最好的精确度
best_model_wts = copy.deepcopy(net.state_dict())
best_acc = 0.0

# 开始周期训练
for epoch in range(num_epoch):
    print('Epoch {}/{}'.format(epoch, num_epoch - 1))
    print('-' * 10)

    # 定义训练的损失，正确率，整体数据量
    train_loss = 0.0
    train_acc = 0.0
    n = 0
    # 统计批次数量,整体数据量
    num_batch = 0
    # 梯度累积,在特定批次之后进行参数的更新，可以缓解因GPU内存不足引起只能设置小批次情况，相当于变相扩大批次大小
    batch_accumulate_size = 4

    # 重置梯度张量
    optimizer.zero_grad()

    # 迭代整个数据集
    for index, data in enumerate(train_data_loader):
        # 获取图像和标签数据
        images, labels = data
        # 若gpu存在，将图像和标签数据放入gpu上
        images = images.to(device)
        labels = labels.to(device)

        # # 将梯度参数设置为0
        # optimizer.zero_grad()

        # 前向传播
        outputs = net(images)
        outputs = F.softmax(outputs, dim=1)

        # 两中预测方法
        preds = torch.argmax(outputs, 1)
        loss = criterion(outputs, labels)
        loss = loss / batch_accumulate_size  # 如果损失在训练时候样本上要进行平均的话，需要除以梯度累积的步骤

        # 仅仅在训练的情况下，进行反向传播，计算梯度值
        loss.backward()

        # 在特定步骤下才进行梯度更新
        # 累积梯度意味着，在调用 optimizer.step() 实施一步梯度下降之前，我们会对 parameter.grad 张量中的几个
        # 反向运算的梯度求和。在 PyTorch 中这一点很容易实现，因为梯度张量在不调用 model.zero_grad() 或
        # optimizer.zero_grad() 的情况下不会重置。如果损失在训练样本上要取平均，我们还需要除以累积步骤的数量。
        if (index + 1) % batch_accumulate_size == 0:
            # 更新参数权重，梯度下降
            optimizer.step()
            # 重置梯度张量
            optimizer.zero_grad()

        # print('loss:{}, accuracy{}'.format(loss, torch.sum(preds == labels.data).double() / images.size(0)))
        # 统计损失,准确值,数据数量
        train_loss += loss.item() * batch_accumulate_size
        train_acc += torch.sum(preds == labels).item()
        # 统计一周期内批次的数量,样本数量
        num_batch += 1
        n += labels.size(0)

    # 每30个周期对学习率进行10倍的衰减
    if (epoch + 1) % 50 == 0:
        learning_rate = learning_rate / 10

    # 计算每周期的损失函数和正确率
    epoch_loss = train_loss / num_batch
    epoch_acc = train_acc / n
    print('Loss: {}, Acc: {}'.format(epoch_loss, epoch_acc))

    # 选出最好的模型参数
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(net.state_dict())
        # 保存最好的模型参数
        torch.save(best_model_wts, model_path)
        print('epoch:{}, update model...'.format(epoch))
    print()

# 训练结束时间
end_time = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(
    end_time // 60, end_time % 60))
print('Best val Acc: {:4f}'.format(best_acc))
