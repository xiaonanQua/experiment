import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.optim import Adam
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import os
import time

# ---------------------配置阶段------------------------------
# 数据集根目录、项目根目录、训练数据保存目录（实验室,根据个人情况设定）
root_dataset = '/home/team/xiaonan/Dataset/'
root_project = '/home/team/xiaonan/experients/'
root_data_save = '/home/team/xiaonan/data_save/'
# 数据集根目录、项目根目录、训练数据保存目录（本机）
# self.root_dataset = '/home/xiaonan/Dataset/'
# self.root_project = '/home/xiaonan/experients/'
# self.root_data_save = '/home/xiaonan/data_save/'
# 数据集根目录、项目根目录、训练数据保存目录（服务器）
# root_dataset = 'Dataset/'
# root_project = ''
# root_data_save = 'data_save/'

# 模型保存目录、日志文件保存目录
model_dir = root_data_save + 'checkpoints/'
log_dir = root_data_save + 'log/'
# 若文件夹不存在，则创建
if os.path.exists(root_data_save) is False:
    os.mkdir(root_data_save)
if os.path.exists(model_dir) is False:
    os.mkdir(model_dir)
if os.path.exists(log_dir) is False:
    os.mkdir(log_dir)

# cifar-10数据集目录；模型名称；类别数量
cifar_10_dir = root_dataset + 'cifar-10/'
model_dir = model_dir + 'cifar10_resnet50_v1' + '.pth'
log_dir = log_dir + 'cifar10_resnet50_v1'
num_classes = 10
if os.path.exists(log_dir) is False:
    os.mkdir(log_dir)

# 检查设备情况
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:{}'.format(device))

# 设置超参数
epochs = 200
batch_size = 32
learning_rate = 0.1
lr_step_size = 30
weight_decay = 1e-4
momentum = 0.9

# 均值和标准差值
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.255]

# -----------------------------读取数据集--------------------------------
# 训练集、验证集、测试集预处理
train_data_preprocess = transforms.Compose([transforms.Resize(size=(224, 224)),
                                            #transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(brightness=0.4, saturation=0.4,
                                                                   hue=0.4, contrast=0.4),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean,
                                                                 std=std)])
valid_data_preprocess = transforms.Compose([transforms.Resize(size=(224, 224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=mean,
                                                                std=std)])
# 获取训练集、测试集
train_dataset = CIFAR10(root=cifar_10_dir, train=True, transform=train_data_preprocess)
test_dataset = CIFAR10(root=cifar_10_dir, train=False, transform=valid_data_preprocess)

# 获取训练集和测试集的加载器
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True, num_workers=4)

# ------------------------构建网络、定义损失函数和优化器------------------------
net = resnet50()
print(net)
# 重写网络的最后一层
fc_in_features = net.fc.in_features  # 网络最后一层的输入通道
print(fc_in_features)
net.fc = nn.Linear(in_features=fc_in_features, out_features=num_classes)
print(net)
# 将网络放置到GPU上
net.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(params=net.parameters(), weight_decay=weight_decay)

# ----------------------进行网络的训练------------------------------------
print('进行训练....')

# 获得记录日志信息的写入器
writer = SummaryWriter(log_dir)


# ------------------定义训练、验证子函数--------------------
# 训练子函数
def _train(train_loader, num_step):
    print('  training stage....')
    # 将网络结构调成训练模式；初始化梯度张量
    net.train()
    optimizer.zero_grad()
    # 定义准确率变量，损失值，批次数量,样本总数量
    train_acc = 0.0
    train_loss = 0.0
    num_batch = 0
    num_samples = 0

    # 进行网络的训练
    for index, data in enumerate(train_loader, start=0):
        # 获取每批次的训练数据、并将训练数据放入GPU中
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        # 推理输出网络预测值，并使用softmax使预测值满足0-1概率范围；计算损失函数值
        outputs = net(images)
        outputs = F.softmax(outputs, dim=1)
        loss = criterion(outputs, labels)

        # 计算每个预测值概率最大的索引（下标）
        preds = torch.argmax(outputs, dim=1)

        # 计算批次的准确率，预测值中预测正确的样本占总样本的比例
        # 统计准确率、损失值、批次数量
        acc = torch.sum(preds == labels).item()
        train_acc += acc
        train_loss += loss
        num_batch += 1
        num_samples += images.size(0)

        # 反向传播（计算梯度）；梯度下降优化（更新参数）；重置梯度张量
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 输出一定次数的损失和精度情况
        if (index + 1) % 30 == 0:
            # 输出损失值和精度值
            print('   batch:{}, batch_loss:{:.4f}, batch_acc:{:.4f}\n'.
                  format(index, loss, acc / images.size(0)))

        # 记录训练批次的损失和准确率
        # writer.add_scalar('Train/Loss', scalar_value=loss, global_step=index)  # 单个标签
        writer.add_scalars(main_tag='Train(batch)',
                           tag_scalar_dict={'batch_loss': loss,
                                            'batch_accuracy': acc / images.size(0)},
                           global_step=num_step)
        # 更新全局步骤
        num_step += 1

    # 计算训练的准确率和损失值
    train_acc = train_acc / num_samples
    train_loss = train_loss / num_batch
    return train_acc, train_loss, num_step


# 验证子函数
def _valid(valid_loader):
    print('  valid stage...')
    # 将网络结构调成验证模式;所有样本的准确率、损失值;统计批次数量;
    net.eval()
    valid_acc = 0.0
    valid_loss = 0.0
    num_batch = 0
    num_samples = 0

    # 进行测试集的测试
    with torch.no_grad():  # 不使用梯度，减少内存占用
        for images, labels in valid_loader:
            # 将测试数据放入GPU上
            images, labels = images.to(device), labels.to(device)
            # 推理输出网络预测值，并使用softmax使预测值满足0-1概率范围
            outputs = net(images)
            outputs = F.softmax(outputs, dim=1)
            # 计算每个预测值概率最大的索引（下标）；计算损失值
            pred = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)

            # 统计真实标签和预测标签的对应情况;计算损失
            valid_acc += torch.sum((pred == labels)).item()
            valid_loss += loss
            num_batch += 1
            num_samples += images.size(0)

    # 计算测试精度和损失值
    valid_acc = valid_acc / num_samples
    valid_loss = valid_loss / num_batch

    return valid_acc, valid_loss


# ----------------------------开始周期训练--------------------------------
# 定义训练开始时间、最好验证准确度（用于保存最好的模型）、统计训练步骤总数
start_time = time.time()
best_acc = 0.0
num_step = 0

# 开始周期训练
for epoch in range(epochs):
    # 设定每周期开始时间点、周期信息
    epoch_start_time = time.time()
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    print('-' * 20)

    # 训练
    train_acc, train_loss, num_step = _train(train_loader, num_step)
    # 验证
    valid_acc, valid_loss = _valid(test_loader)

    # 输出每周期的训练、验证的平均损失值、准确率
    epoch_time = time.time() - epoch_start_time
    print('   epoch：{}/{}, time:{:.0f}m {:.0f}s'.
          format(epoch, epochs, epoch_time // 60, epoch_time % 60))
    print('   train_loss:{:.4f}, train_acc:{:.4f}\n   valid_loss:{:.4f}, valid_acc:{:.4f}'.
          format(train_loss, train_acc, valid_loss, valid_acc))

    # 记录测试结果
    writer.add_scalars(main_tag='Train(epoch)',
                       tag_scalar_dict={'train_loss': train_loss, 'train_acc': train_acc,
                                        'valid_loss': valid_loss, 'valid_acc': valid_acc},
                       global_step=epoch)

    # 选出最好的模型参数
    if valid_acc > best_acc:
        # 更新最好精度、保存最好的模型参数
        best_acc = valid_acc
        torch.save(net.state_dict(), model_dir)
        print('  epoch:{}, update model...'.format(epoch))
    print()

# 训练结束时间、输出最好的精度
end_time = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))
print('Best val Acc: {:4f}'.format(best_acc))

# 关闭writer
writer.close()
