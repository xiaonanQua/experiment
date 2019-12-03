import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from config.config import Config

# -----------------------读取cifar-10数据集--------------------------
# 实例化配置文件
cfg = Config()
device = 'cuda:0' if torch.cuda.is_available()  else 'cpu'

# 读取cifar-10数据集
dataset = CIFAR10(root=cfg.cifar_10_dir, transform=transforms.ToTensor(), download=True)
# 数据加载器
data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

# 保存部分图像
for index, data in enumerate(data_loader):
    # 获得批次图片数据及批次大小
    images, _ = data
    batch_size = images.size(0)
    print('#{} has {} images'.format(index, batch_size))

    # 每100次进行一次保存
    if index % 100 == 0:
        # 保存路径
        path = '../result/gan_save_image/cifar10_batch_{:03d}.png'.format(index)
        save_image(images, filename=path, normalize=True)

# ----------------------搭建生成网络和鉴别网络---------------------------------------------
# 搭建生成网络
latent_size = 64  # 潜在大小
n_channel = 3  # 输出通道数
n_g_feature = 64  # 生成网络隐藏层大小
g_net = nn.Sequential(
    # 第一层：输入大小=（64,1,1）,应用转置卷积，批次规范化，ReLU激活
    nn.ConvTranspose2d(in_channels=latent_size, out_channels=4*n_g_feature, kernel_size=4, bias=False),
    nn.BatchNorm2d(4*n_g_feature),
    nn.ReLU(),
    # 大小=（256,4,4）
    nn.ConvTranspose2d(in_channels=4*n_g_feature, out_channels=2*n_g_feature, kernel_size=4, stride=2,
                       padding=1, bias=False),
    nn.BatchNorm2d(2*n_g_feature),
    nn.ReLU(),
    # 大小=（128,8,8）
    nn.ConvTranspose2d(in_channels=2*n_g_feature, out_channels=n_g_feature, kernel_size=4, stride=2,
                       padding=1, bias=False),
    nn.BatchNorm2d(n_g_feature),
    nn.ReLU(),
    # 大小=（64,16,16）
    nn.ConvTranspose2d(n_g_feature, n_channel, kernel_size=4, stride=2, padding=1),
    nn.Sigmoid()
    # 图片大小（3,32,32）
)
print('生成网络结构...\n', g_net)

# 搭建鉴别网络
n_d_feature = 64  # 鉴别网络隐藏层大小
d_net = nn.Sequential(
    # 图片大小=（3,32,32）
    nn.Conv2d(n_channel, n_d_feature, kernel_size=4, stride=2, padding=1),
    nn.LeakyReLU(0.2),
    # 大小=（64,16,16）
    nn.Conv2d(n_d_feature, 2*n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2*n_d_feature),
    nn.LeakyReLU(0.2),
    # 大小=（128,8,8）
    nn.Conv2d(2*n_d_feature, 4*n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(4*n_d_feature),
    nn.LeakyReLU(0.2),
    # 大小=(256, 4, 4)
    nn.Conv2d(4*n_d_feature, 1, kernel_size=4)
    # 对数赔率张量大小（1,1,1）
)
print('鉴别网络结构...\n', d_net)

# ------------------------对网络权重值进行初始化-----------------------------
def weight_init(m):
    # 若权重类型是转置卷积和互相关卷积，则初始化Xavier的正太分布
    if type(m)  in [nn.ConvTranspose2d, nn.Conv2d]:
        init.xavier_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:  # 若是批次规范化，则初始化权重为标准正太分布，偏差为常数
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)
# 对生成网络和鉴别网络的权重进行初始化
g_net.apply(weight_init)
d_net.apply(weight_init)
# 将网络结构置于gpu上
g_net.to(device)
d_net.to(device)

# ------------------------对网络进行训练-------------------------------------

# 实例化损失对象
criterion = nn.BCEWithLogitsLoss()
# 定义生成网络和鉴别网络的优化器
g_optimizer = optim.Adam(g_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(d_net.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 设置测试的固定噪声，用来查看相同的潜在张量在训练过程中生成图片的变换
batch_size = 64
fixed_noises = torch.randn(batch_size, latent_size, 1, 1)

# 训练过程
epoch_num = 15  # 训练周期
for epoch in range(epoch_num):
    for index, data in enumerate(data_loader):
        # 载入本批次数据
        real_images, _ = data
        real_images = real_images.to(device)
        batch_size = real_images.size(0)  # 批次大小

        # 训练鉴别网络
        # 对真数据进行鉴别训练
        labels = torch.ones(batch_size)  # 真实数据对应标签为1
        labels = labels.to(device)
        preds = d_net(real_images)  # 对真实数据进行判别
        outputs = preds.reshape(-1)
        d_loss_real = criterion(outputs, labels)  # 真是数据的鉴别器损失
        d_mean_real = outputs.sigmoid().mean()  # 计算鉴别器将多少比例的真数据判定为真，仅用于输出

        # 对假数据进行鉴别训练
        noises = torch.randn(batch_size, latent_size, 1, 1)  # 潜在噪声
        fake_images = g_net(noises)  # 生成假数据
        labels = torch.zeros(batch_size) # 假数据对应标签为0
        labels = labels.to(device)
        fake = fake_images.detach()  # 使得梯度的计算不回溯到生成网络，加快训练速度。删掉此步，不影响结果
        preds = d_net(fake)  # 对假数据进行鉴别
        outputs = preds.reshape(-1)
        d_loss_fake = criterion(outputs, labels)  # 假数据的鉴别损失
        d_mean_fake = outputs.sigmoid().mean()   # 计算鉴别器将多少比例的假数据判定为真，仅用于输出

        # 计算总的鉴别损失，进行反向传播计算和参数的更新
        d_loss = d_loss_real + d_loss_fake  # 总的鉴别损失
        d_net.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # 训练生成网络
        labels = torch.ones(batch_size)  # 生成网络希望所有生成的数据都被认为是真数据
        labels = labels.to(device)
        preds = d_net(fake_images)  # 让假数据通过鉴别网络
        outputs = preds.view(-1)
        g_loss = criterion(outputs, labels)  # 从真数据看到的损失
        g_mean_fake = outputs.sigmoid().mean()  # 计算鉴别器将多少比例的假数据判定为真，仅用于输出

        # 计算总的鉴别损失，进行反向传播计算和参数的更新
        g_net.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # 输出本批次的训练结果
        print('[{}/{}]'.format(epoch, epoch_num) +
              '[{}/{}]'.format(index, len(data_loader)) +
              '鉴别网络损失：{:g} 生成网络损失：{:g}'.format(d_loss, g_loss) +
              '真数据判真比例：{:g} 假数据判真比例：{:g}/{:g}'.format(d_mean_real, d_mean_fake, g_mean_fake)
              )

        # 经过一定批次保存生成网络生成的图片
        if index % 100 == 0:
            fake = g_net(fixed_noises)  # 由固定潜在张量生成假数据
            # 保存路径
            path = '../result/gan_save_image/epoch{:02d}_batch_{:03d}.png'.format(epoch, index)
            save_image(fake, path)




