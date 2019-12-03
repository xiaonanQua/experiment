"""
训练一个分类器
"""

import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pytorch_learn.neural_network import LeNet


# 配置数据集存放根目录
cifar_path = '/home/xiaonan/Dataset/cifar-10/'

# 超参数配置
batch_size = 6
learning_rate = 0.001
momentum = 0.9
num_epochs = 4

# 开启线程数
num_threads = 4

# 定义对数据的预处理
data_preprocess = transforms.Compose([
    transforms.ToTensor(),  # 转化成Tensor,使数据归一化在0-1范围内
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   # 使数据归一化为均值标准差都为0.5
])

# 获取训练集、测试集
trainset = datasets.CIFAR10(root=cifar_path,
                            train=True,
                            transform=data_preprocess,
                            download=True)

testset = datasets.CIFAR10(root=cifar_path,
                           train=False,
                           transform=data_preprocess,
                           download=True)

# 训练、测试集的数据加载器
train_loader = DataLoader(dataset=trainset,
                          batch_size=batch_size,
                          shuffle=True,  # 进行洗牌操作
                          num_workers=num_threads  # 开启多线程，线程数
                          )
test_loader = DataLoader(dataset=testset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=num_threads)

# 显示图像的函数
def imshow(image):
    image = image/2 + 0.5
    image = image.numpy()  # 将Tensor转化成numpy
    plt.imshow(np.transpose(image, (1, 2, 0)))  # 将图像的格式转化成[高，宽，颜色通道]
    plt.show()


# 定义类别元组
classes = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 通过下标访问数据集的数据
(image, label) = trainset[10]
print('第10个图片的类别是:', classes[label])
print('打印图片...')
imshow(image)

# DataLoader对象是一个可迭代的对象，它将dataset返回的每一条数据样本拼接成一个batch，并提供多线程加速优化和数据打乱等操作。
dataiter = iter(train_loader)  # 获得训练数据加载器的迭代器
images, labels = dataiter.__next__()  # 返回6张图片及标签
print(''.join('%11s'%classes[label] for label in range(batch_size)))
imshow(torchvision.utils.make_grid(images))

# 获得网络结构
net = LeNet()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(params=net.parameters(),
                      lr=learning_rate,
                      momentum=momentum)

# 训练网络
for epoch in range(num_epochs):
    # 汇总运行损失
    running_loss = 0.0
    for i,data in enumerate(train_loader, start=0):
        # 获得训练图像和标签，data是一个列表[images,labels]
        images, labels = data

        # 将参数梯度设置为0
        optimizer.zero_grad()

        # 进行前向传播，反向传播，优化参数
        logit = net(images)
        loss = criterion(logit, labels)
        loss.backward()
        optimizer.step()  # 更新参数

        # 输出统计值
        running_loss += loss
        if (i+1)%2000 == 0:
            print('[%d, %5d]loss:%.3f'%(epoch+1, i+1, running_loss/2000))
            running_loss = 0.0
print('训练完成...')

# 获得测试集加载器的迭代器
test_iter = iter(test_loader)
images, labels = test_iter.__next__()

# 输出图片
imshow(torchvision.utils.make_grid(tensor=images))
print('图片类别：',''.join('%5d'%classes[label] for label in range(batch_size)))

print()
# 计算图片在每个类别上的分数
outputs = net(images)
_, predicted = torch.max(outputs, 1)  # 寻找出每一行的最大值
print('用测试集预测的结果：', ''.join('%5s'%classes[predicted[j]] for j in range(batch_size)))

# 计算整个测试集的准确率
correct = 0  # 预测正确的图片数
total = 0  # 总共的图片数

# 禁用梯度计算的上下文管理器，减少内存计算
with torch.no_grad():
    for images, labels in test_loader:
        # 训练模型预测的结果
        outputs = net(images)
        # 找出每一样本预测中得分最高的值
        _,predicted = torch.max(outputs, 1)
        # 将批次样本数量加入统计样本数量中
        total += labels.size(0)
        # 将预测的类别值和真实类别值对比，计算准确率
        correct += (predicted==labels).sum().item()
print('{}张测试集中的准确率：%d %%'%(100*correct/total))

# 计算每个类别的准确率
class_correct = list(0. for i in range(10))  # 统计每个类别的预测正确的样本数
class_total = list(0. for i in range(10))  # 统计每个类别的样本数量

with torch.no_grad():
    for images, labels in test_loader:
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()  # 使用squeeze()将预测标签中形状是1移除掉
        for i in range(batch_size):
            class_correct[label] += c[i].item()
            class_total[label] += 1

# 输出所有类别的准确率
for i in range(10):
    print('Accuracy of %5s：%2d %%'%(classes[i], 100*class_correct[i]/class_total[i]))

# 在GPU上训练
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)
net.to(device)



