# -*- coding:utf-8 -*-
from __future__ import division, print_function, absolute_import
import math
import sys
import os
import time
import torch
from torch.utils.data import random_split
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from config.cifar10_config import Cifar10Config
import cv2
import zipfile
import random


def view_bar(message, num, total):
    """
    进度条工具
    :param message: 进度条信息
    :param num: 当前的值,从1开始..
    :param total: 整体的值
    :return:
    """
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">"*rate_num, ""*(40-rate_num), rate_nums, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()


def mkdir(dir_path):
    """
    判断文件夹是否存在,创建文件夹
    :param dir_path: 文件夹路径
    :return:
    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def imshow(images, title=None):
    """
    显示一些PIL格式的图像
    :param images: 图像
    :param title: 标题
    :return:
    """
    # 将PIL格式的图像转化成numpy形式，再将图像维度转化成（高度，宽度，颜色通道）
    images = images.numpy().transpose([1, 2, 0])
    # 设置平均值和标准差
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # 将进行归一化后的图像还原
    images = images * std + mean
    images = np.clip(images, 0, 1)
    plt.imshow(images)
    if title is not None:
        plt.title(title)

def show_image(images, num_rows, num_cols, scale=2):
    """
    显示多个图片
    :param images: 多个图片
    :param num_rows: 行数量
    :param num_cols: 列数量
    :param scale: 尺度
    :return:
    """
    # 图像大小
    figsize = (num_cols*scale, num_rows*scale)


def one_hot_embedding(labels, num_classes):
    """
    将标签嵌入成one-hot形式
    :param labels: 标签,（LongTensor）类别标签,形状[N,]
    :param num_classes: 类别数,
    :return:(tensor)被编码的标签,形状（N,类别数）
    """
    # 返回2维张量，对角线全是1，其余全是0
    y = torch.eye(num_classes)
    return y[labels]  # 使用按行广播机制


def split_valid_set(dataset, save_coef):
    """
    从原始数据集中划分出一定比例的验证集
    :param dataset: 原始数据集，一般是训练集。这里的数据集是经过pytorch中DataSet读取出来的数据集对象。
    :param save_coef: 保存原始数据集的系数
    :return: 划分后的数据集。格式类似于：train_dataset, valid_dataset
    """
    # 训练集的长度
    train_length = int(save_coef*len(dataset))
    # 验证集的长度
    valid_length = len(dataset) - train_length
    # 使用pytorch中的随机划分成数据集来划分
    train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length])

    return train_dataset, valid_dataset


def show_label_distribute(data_loader):
    """
    显示数据集标签的分布情况
    :param data_loader: 数据集加载器（pytorch加载器对象）
    :return:
    """
    print('label distribution ..')
    figure, axes = plt.subplots()
    labels = [label.numpy().tolist() for _, label in data_loader]
    print(labels)
    class_labels, counts = np.unique(labels, return_counts=True)
    axes.bar(class_labels, counts)
    axes.set_xticks(class_labels)
    plt.show()


def vis(test_accs, confusion_mtxes, labels, figsize=(20, 8)):
    cm = confusion_mtxes[np.argmax(test_accs)]
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%' % p
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'

    fig = plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.plot(test_accs, 'g')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    sn.heatmap(cm, annot=annot, fmt='', cmap="Blues")
    plt.show()


def confusion_matrix(targets, preds):
    """
    生成混淆矩阵
    :param targets: 真实标签数据，数据格式list
    :param preds: 与真实标签对应的预测标签数据，数据格式list
    :return: 混淆矩阵
    """
    # 统计真实标签中的类别数量
    num_class = len(set(targets))
    # 初始化相应类别数量大小的混淆矩阵
    conf_matrix = np.zeros(shape=[num_class, num_class])
    print(conf_matrix)
    # 判断真实标签与预测标签的数量是否相等
    if len(targets) != len(preds):
        raise Exception('The number of real and predicted labels is inconsistent')
    # 进行标签的统计
    for i in range(len(targets)):
        true_i = np.array(targets[i])
        pred_i = np.array(preds[i])
        conf_matrix[true_i, pred_i] += 1.0

    return conf_matrix


def visiual_confusion_matrix(confusion_mat, classes_name, graph_name=None, out_path=None):
    """
    可视化混淆矩阵
    :param confusion_mat: 统计好的混淆矩阵
    :param classes_name: 混淆矩阵对应的类别名称
    :param graph_name: 当前图的名称
    :param out_path: 以png的图像格式保存混淆矩阵
    :return:
    """
    # 判断混淆矩阵中的类别与类别名称中类别数量是否一致
    if confusion_mat.shape[0] != len(classes_name):
        raise Exception('Inconsistent number of categories')
    # 对混淆矩阵逐行进行数值归一化
    confusion_mat_normal = confusion_mat.copy()
    for i in range(len(classes_name)):
        confusion_mat_normal[i, :] = confusion_mat[i, :] /confusion_mat_normal[i, :].sum()
    print(confusion_mat_normal)

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')
    plt.imshow(confusion_mat_normal, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes_name)))
    plt.xticks(xlocations, classes_name, rotation=60)
    plt.yticks(xlocations, classes_name)
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('' + graph_name)

    # 打印数字
    for i in range(confusion_mat_normal.shape[0]):
        for j in range(confusion_mat_normal.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    if out_path is not None:
        plt.savefig(os.path.join(out_path, 'Confusion_Matrix_' + graph_name + '.png'))
    plt.show()
    plt.close()


def read_and_write_videos(video_files=None, out_files=None):
    """
    通过OpenCV中的VideoCapture函数调用系统摄像头读取视频图像，或者读取特定视频文件
    :param video_files: 读取的视频文件地址，若为Ｎｏｎｅ则读取摄像头文件
    :param out_files: 输出文件
    :return:
    """
    # 创建VideoCapture进行一帧一帧视频读取
    if video_files is None:
        # 调用系统单个摄像头作为视频输入
        cap = cv2.VideoCapture(0)
    else:
        # 读取特定视频文件
        cap = cv2.VideoCapture(video_files)

    # 判断摄像头是否打开
    if cap.isOpened() is False:
        print('Error opening video stream or file')

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(out_files,
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          10, (frame_width, frame_height))

    # 读取视频，直到读取所有时间段视频
    while cap.isOpened():
        # 一帧一帧的读取视频
        ret, frame = cap.read()
        if ret == True:
            out.write(frame)
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, 'xiaonan', (30, 30), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'xiaoshuai', (30, 90), font, 1, (0, 0, 255), 2)

            # 显示帧结果
            cv2.imshow('frame', frame)
            # 播放每一帧时等待25秒或者按ｑ结束
            if cv2.waitKey(1)&0xFF==ord('q'):
                print('结束..')
                break
        else:  # 结束循环
            break

    # 当视频读取结束时，释放视频捕捉的输出Ｏ
    cap.release()
    # 关闭所有帧窗口
    cv2.destroyAllWindows()


def read_jay_lyrics(num_examples=None):
    """
    读取周杰伦歌词数据集
    :param samples: 设置读取的样本数
    :return:
    """
    # 打开周杰伦歌词文件
    file_path = '/home/xiaonan/Dataset/lyrics/jaychou_lyrics.txt.zip'
    with zipfile.ZipFile(file=file_path) as zin:
        with zin.open('jaychou_lyrics.txt') as file:
            lyrics = file.read().decode('utf-8')
    lyrics = lyrics.replace('\n', ' ').replace('\r', ' ')
    if num_examples is not None:
        train_set = lyrics[:num_examples]
    else:
        train_set = lyrics

    # 建立字符索引
    idx_to_char = list(set(train_set))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    # 将训练集字符转化成索引
    train_set = [char_to_idx[char] for char in train_set]

    return train_set, vocab_size, idx_to_char, char_to_idx


def seq_random_sample(samples_indices, batch_size, num_steps, device=None):
    """
    时序数据的随机采样
    :param samples_indices:样本数据的索引
    :param batch_size: 批次大小，每个小批量的样本数
    :param num_steps: 每个样本所包含的时间步数
    :param device: 数据采样放置在什么设备上
    :return:
    """
    # 减１是因为输出的索引ｘ是相应输入的索引ｙ加１
    num_examples = (len(samples_indices) -1)//num_steps
    # 周期大小
    epoch_size = num_examples // batch_size
    # 样本索引
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)
    #　放置设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(epoch_size):
        # 每次读取batch_size个的样本
        i = i*batch_size
        batch_indices = example_indices[i:i+batch_size]
        X = [samples_indices[j*num_steps: j*num_steps+num_steps] for j in batch_indices]
        Y = [samples_indices[j*num_steps+1: j*num_steps+1+num_steps] for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), \
              torch.tensor(Y, dtype=torch.float32, device=device)


def seq_adjacent_sample(example_indices, batch_size, num_steps, device=None):
    # 获取设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 将索引数据转化成张量属性
    example_indices = torch.tensor(example_indices, dtype=torch.float32,
                                   device=device)
    # 序列数据的长度
    data_len = example_indices.size(0)
    # 批次数量
    num_batch = data_len // batch_size
    # 转化索引数据成(批次大小，批次长度)格式
    indices = example_indices[0: batch_size*num_batch].view(batch_size, num_batch)
    # 计算周期大小
    epoch_size = (num_batch-1)//num_steps
    for i in range(epoch_size):
        i = i*num_steps
        x = indices[:, i:i+num_steps]
        y = indices[:, i+1:i+num_steps+1]
        yield x, y


def _one_hot(seq_data, vocab_size, dtype=torch.float32):
    """
    将序列数据转化成one-hot向量,即转成词向量。
    :param seq_data:　序列数据的索引，格式：[batch_size]-->[batch_size, vocab_size]
    :param vocab_size: 序列数据中不同词的数量
    :param dtype:　数据类型,默认ＦＬＯＡＴ３２
    :return:
    """
    x = seq_data.long()
    res = torch.zeros(seq_data.shape[0], vocab_size, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


def seq_one_hot(seq_data, vocab_size):
    """
    将序列数据转化成one-hot向量,即转成词向量。
    :param seq_data:　序列数据的索引，格式：[batch_size, seq_len]-->序列长度个[batch_size, vocab_size]
    :param vocab_size: 序列数据中不同词的数量
    :return:
    """
    return [_one_hot(seq_data[:, i], vocab_size) for i in range(seq_data.shape[1])]


def rnn_predict(prefix, num_chars, rnn, params, init_rnn_state, num_hiddens,
                vocab_size, device, idx_to_char, char_to_idx):
    """
    根据一段前缀的词进行预测
    :param prefix:　预测的词
    :param num_chars:字符数量
    :param rnn:rnn函数
    :param params:初始化的参数
    :param init_rnn_state:初始化隐状态
    :param num_hiddens:隐藏单元的数量
    :param vocab_size:　不同字典的数量
    :param device:　设备
    :param idx_to_char:　根据索引找字符
    :param char_to_idx:　根据字符找索引
    :return:
    """
    # 初始化隐状态
    state = init_rnn_state(1, num_hiddens, device)
    # 输出
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars+len(prefix)-1):
        # 将上一时间步的输出作为当前时间步的输入
        X = seq_one_hot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix)-1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))

    return ''.join([idx_to_char[i] for i in output])


def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data **2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta/norm)

def rnn(inputs, state, params):
    """
    在一个时间步里如何计算隐藏状态和输出。
    input和output都是num_steps时间步个形状为(batch_size, vocab_size)的词向量
    :param inputs:当次输入数据
    :param state:隐状态
    :param params:参数
    :return:(当前层的输出,隐状态)
    """
    # 获取初始的参数、上一时刻的隐藏
    W_xh, W_hh, b_h, W_hq, b_q = params()
    H, = state
    outputs = []
    for X in inputs:
        # 隐状态
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        # 输出
        Y = torch.matmul(H, W_hq) + b_q
        #　保存输出
        outputs.append(Y)
    return outputs, (H,)


# 初始化隐藏数量
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def sgd(params, lr, batch_size):
    """
    根据计算的梯度更新参数
    :param params: 需更新的参数
    :param lr: 学习率
    :param batch_size: 批次大小
    :return:
    """
    for param in params:
        param.data -= lr*param.grad / batch_size





if __name__ == "__main__":
    # labels = torch.tensor([1, 2, 3, 1])
    # # print(labels.squeeze(1))
    # print(one_hot_embedding(labels, 4))
    # cfg = Cifar10Config()
    # test_loader = cfg.dataset_loader(cfg.cifar_10_dir, train=False, shuffle=False)
    # show_label_distribute(test_loader)
    # video_file = '/home/xiaonan/sf6_1.avi'
    # read_and_write_videos()
    # my_seq = list(range(30))
    # for X,Y in seq_adjacent_sample(my_seq, batch_size=3, num_steps=5):
    #     print(X,Y)
    x = torch.arange(10).view(2, 5)
    print(x)
    inputs = seq_one_hot(x, 2045)
    print(len(inputs), inputs[0].size())