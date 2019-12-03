import torch
import torch.nn.functional as F
import torchvision
import time
import copy
import os
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.nn import DataParallel
from utils.tools import confusion_matrix, visiual_confusion_matrix
import numpy as np


def train_and_valid_(net, criterion, optimizer, train_loader, valid_loader, cfg,
                     is_lr_adjust=True, is_lr_warmup=False):

    # ------------------配置信息------------------------------
    # 若检查点存在且容许使用检查点，则加载参数进行训练
    if os.path.exists(cfg.checkpoints) and cfg.use_checkpoints:
        # 加载权重信息
        net.load_state_dict(torch.load(cfg.checkpoints))
        print('加载权重信息...')

    # 配置学习率衰减器(默认是按epoch衰减);两种类型的学习率衰减
    if is_lr_adjust:
        # 按一定周期之后进行衰减<StepLR>
        lr_shcleduler_step = StepLR(optimizer=optimizer, step_size=cfg.lr_decay_step)
    elif is_lr_warmup:  # 若True，则开启学习率预热
        # 定义Lambda表达式 < LambdaLR >
        lr_lambda = lambda epoch: epoch / cfg.lr_warmup_step
        lr_shcleduler_warmup = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
        lr_shcleduler_warmup.step()

    # 获得记录日志信息的写入器
    writer = SummaryWriter(cfg.log_dir)

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
            # print(images.size(), labels)
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)

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

            # 判断是否使用梯度累积技巧（显存少的时候），否则，进行正常的反向传播（计算梯度）和梯度下降优化操作
            if cfg.grad_accuml is True and cfg.batch_size < 128:
                # 累积损失，求累积损失的平均损失
                loss = loss/cfg.batch_accumulate_size
                loss.backward()
                # 满足一定批次要求则进行梯度参数更新，重置梯度张量
                if (index + 1) % cfg.batch_accumulate_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                # 计算梯度、更新参数、重置梯度张量
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # 输出一定次数的损失和精度情况
            if (index+1) % cfg.print_rate == 0:
                # 输出损失值和精度值
                print('   batch:{}, batch_loss:{:.4f}, batch_acc:{:.4f}\n'.
                      format(index, loss, acc/images.size(0)))

            # 记录训练批次的损失和准确率
            # writer.add_scalar('Train/Loss', scalar_value=loss, global_step=index)  # 单个标签
            writer.add_scalars(main_tag='Train(batch)',
                               tag_scalar_dict={'batch_loss': loss,
                                                'batch_accuracy': acc/images.size(0)},
                               global_step=num_step)
            # 更新全局步骤
            num_step += 1

        # 计算训练的准确率和损失值
        train_acc = train_acc/num_samples
        train_loss = train_loss/num_batch
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
            for index, data in enumerate(valid_loader, start=0):
                images, labels = data
                # 将测试数据放入GPU上
                images, labels = images.to(cfg.device), labels.to(cfg.device)
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
        valid_acc = valid_acc/num_samples
        valid_loss = valid_loss / num_batch

        return valid_acc, valid_loss

    # ----------------------------开始周期训练--------------------------------
    # 定义训练开始时间、最好验证准确度（用于保存最好的模型）、统计训练步骤总数
    start_time = time.time()
    best_acc = 0.0
    num_step = 0

    # 开始周期训练
    for epoch in range(cfg.epochs):
        # 设定每周期开始时间点、周期信息
        epoch_start_time = time.time()
        print('Epoch {}/{}'.format(epoch, cfg.epochs - 1))
        print('-' * 20)

        # 训练
        train_acc, train_loss, num_step = _train(train_loader, num_step)
        # 验证
        valid_acc, valid_loss = _valid(valid_loader)

        # 调整学习率
        # 在前几周期内，进行学习率预热
        if is_lr_warmup is True and epoch < cfg.lr_warmup_step:
            lr_shcleduler_warmup.step()
            print('  epoch:{}/{}, learning rate warmup...{}'.
                  format(epoch, cfg.lr_warmup_step - 1, lr_shcleduler_warmup.get_lr()))
        elif is_lr_adjust:  # 在经过一定学习率预热后，学习率恢复成初始的值。或则直接进行周期下降。
            lr_shcleduler_step.step()

        # 输出每周期的训练、验证的平均损失值、准确率
        epoch_time = time.time() - epoch_start_time
        print('   epoch：{}/{}, time:{:.0f}m {:.0f}s'.
              format(epoch, cfg.epochs, epoch_time // 60, epoch_time % 60))
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
            torch.save(net.state_dict(), cfg.checkpoints)
            print('  epoch:{}, update model...'.format(epoch))
        print()

    # 训练结束时间、输出最好的精度
    end_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 关闭writer
    writer.close()


# 测试函数
def test(net, test_loader, cfg):
    print('test stage...\n')
    # 加载模型权重、将网络放入GPU
    if os.path.exists(cfg.checkpoints):
        net.load_state_dict(torch.load(cfg.checkpoints))
        print('load model argument...')
    net.to(cfg.device)

    # 将网络结构调成验证模式、定义准确率、标签列表和预测列表
    net.eval()
    correct = 0
    targets, preds = [], []

    # 进行测试集的测试
    with torch.no_grad():  # 不使用梯度，减少内存占用
        for images, labels in test_loader:
            # 将测试数据放入GPU上
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            # 推理输出网络预测值，并使用softmax使预测值满足0-1概率范围
            outputs = net(images)
            outputs = F.softmax(outputs, dim=1)
            pred = torch.argmax(outputs, dim=1)

            # 计算每个预测值概率最大的索引（下标）；统计真实标签和对应预测标签
            correct += torch.sum((pred == labels)).item()
            targets += list(labels.cpu().numpy())
            preds += list(pred.cpu().numpy())

    # 计算测试精度和混淆矩阵
    test_acc = 100. * correct / len(test_loader.dataset)
    # confusion_mat = metrics.confusion_matrix(targets, preds)
    confusion_mat = confusion_matrix(targets, preds)
    print('numbers samples:{}, test accuracy:{},\nconfusion matrix:\n{}'.
          format(len(test_loader.dataset), test_acc, confusion_mat))
    return test_acc, confusion_mat


def train(model, train_data_loader, criterion, optimizer, cfg, valid_data_loader=None,):
    """
    训练器
    :param model: 网络结构（模型）
    :param train_data_loader: 训练数据集加载器， 加载批次数据
    :param valid_data_loader: 验证数据集，加载整体数据
    :param criterion: 评估器，实例化评估对象
    :param optimizer: 优化器，实例化优化对象
    :param cfg: 配置文件
    :return:
    """
    # 记录训练的开始时间
    start_time = time.time()

    # 若模型存在，则导入模型
    # if os.path.exists(model.model_path):
    #     model.load(model.model_path)
    # 让模型使用gpu
    if cfg.use_gpu:
        model.cuda()

    print('_'*5 + '进行训练...' + '\n')
    # 训练
    for epoch in range(cfg.epochs):
        # 对每个批次的数据进行处理
        for i, data in enumerate(train_data_loader, start=0):
            # 获得训练图像和标签，data是一个列表[images,labels]
            images, labels = data
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)
            # # 使用gpu
            # if cfg.use_gpu:
            #     images = images.cuda()
            #     labels = labels.cuda()

            # 将参数梯度设置为0
            optimizer.zero_grad()

            # 进行前向传播，反向传播，优化参数
            logit = model(images)
            batch_loss = criterion(logit, labels)
            batch_loss.backward()
            optimizer.step()  # 更新参数

            # 返回预测样本中的最大值的索引
            predict = torch.argmax(logit, dim=1)
            # 计算预测样本类别与真实标签的正确值数量
            num_correct = (predict == labels).sum().item()
            # 计算准确率
            batch_accuracy = num_correct/labels.size(0)

            # 获得验证集结果
            # valid_accuracy, valid_loss = val(model=model, dataloader=val_data_loader,
            #                                  criterion=criterion)

            # 输出训练结果
            print('epoch:{},step:{}, batch_loss:{}, batch_accuracy:{}, valid_loss:{}, valid_accuracy:{}'.
                  format(epoch + 1, i + 1, batch_loss, batch_accuracy, None, None))
    end_time = time.time() - start_time
    print('训练完成,训练时间：{:.0f}m {:.0f}s'.format(start_time//60, end_time%60))
    return model.save()  # 保存模型


def train_model(model, dataloaders, criterion, optimizer, scheduler, cfg):
    """
    训练模型，包含验证集
    :param model: 网络结构
    :param dataloaders: 数据加载器，包含训练集和验证集
    :param criterion: 评估器
    :param optimizer: 优化器
    :param scheduler: 学习率调度器
    :param cfg: 配置文件
    :return:
    """
    # 训练的开始时间
    since = time.time()

    # 深层复制模型的状态字典（模型的参数）， 定义最好的精确度
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 开始周期训练
    for epoch in range(cfg.epochs):
        print('Epoch {}/{}'.format(epoch, cfg.epochs - 1))
        print('-' * 10)

        # 每个周期要进行训练和验证两个任务
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为验证模式

            # 定义运行时训练的损失和正确率
            running_loss = 0.0
            running_corrects = 0
            # 统计数据数量
            num_data = 0

            # 迭代整个数据集
            for index, data in dataloaders[phase]:
                # 获取图像和标签数据
                images, labels = data
                # 若gpu存在，将图像和标签数据放入gpu上
                images = images.to(cfg.device)
                labels = labels.to(cfg.device)

                # 将梯度参数设置为0
                optimizer.zero_grad()

                # 前向传播
                # 追踪训练的历史,通过上下文管理器设置计算梯度的开关
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 仅仅在训练的情况下，进行反向传播，更新权重参数
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失,准确值,数据数量
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_data += images[0]

            epoch_loss = running_loss / num_data
            epoch_acc = running_corrects.double() / num_data

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 选出最好的模型参数
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 保存最好的模型参数
    model.state_dict = best_model_wts
    return model.save()  # 保存模型的路径


def val(model, dataloader, criterion=None):
    """
    计算模型在验证集上的准确率等信息
    :param model: 定义的网络模型对象
    :param dataloader: 数据加载器
    :param criterion: 损失函数
    :return:
    """
    # 将训练模式切换成验证模式，因为在验证时对于使用dropout和BatchNorm不需要设置
    model.eval()

    # 将模型切换到cpu上
    device = torch.device('cpu')
    model.to(device)
    # model.cuda()
    # 定义预测样本正确数,整体损失函数值,平均损失值和样本数
    num_correct = 0
    total_loss = 0
    average_loss = 0
    num_total = 0

    # 进行样本验证
    for index, (images, labels) in enumerate(dataloader, start=0):
        # 使用gpu
        # if cfg.use_gpu:
        #     images = images.cuda()
        #     labels = labels.cuda()
        # 获得神经网络的预测值
        logits = model(images)
        # 返回一个张量在特定维度上的最大值的索引
        predicted = torch.argmax(logits, dim=1)
        # 统计批次样本的数量
        num_total += labels.size(0)
        # 统计预测正确样本的值
        num_correct += (predicted == labels).sum().item()

        if criterion is not None:
            # 计算验证样本的损失值并加入整体损失中
            loss = criterion(logits, labels)
            total_loss += loss

    # 计算验证样本的准确率,平均损失
    accuracy = num_correct/num_total
    if criterion is not None:
        average_loss = total_loss/num_total
    # 将训练模式改成训练模式
    model.train()

    return accuracy, average_loss


def train_and_valid(net, criterion, optimizer, train_data_loader, cfg):
    # 若检查点存在且容许使用检查点，则加载参数进行训练
    if os.path.exists(cfg.checkpoints) and cfg.use_checkpoints:
        # 加载权重信息
        net.load_state_dict(torch.load(cfg.checkpoints))
        print('加载权重信息....')

    # 将网络结构、损失函数放置在GPU上
    net.to(cfg.device)
    criterion = criterion.cuda()
    # 配置优化器
    optimizer = optimizer(params=net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    # 训练的开始时间
    start_time = time.time()

    # 深层复制模型的状态字典（模型的参数）， 定义最好的精确度
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0

    # 开始周期训练
    for epoch in range(cfg.epochs):
        print('Epoch {}/{}'.format(epoch, cfg.epochs - 1))
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
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)

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
            cfg.learning_rate = cfg.learning_rate / 10

        # 计算每周期的损失函数和正确率
        epoch_loss = train_loss / num_batch
        epoch_acc = train_acc / n
        print('Loss: {}, Acc: {}'.format(epoch_loss, epoch_acc))

        # 选出最好的模型参数
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(net.state_dict())
            # 保存最好的模型参数
            torch.save(best_model_wts, cfg.checkpoints)
            print('epoch:{}, update model...'.format(epoch))
        print()

    # 训练结束时间
    end_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        end_time // 60, end_time % 60))
    print('Best val Acc: {:4f}'.format(best_acc))







