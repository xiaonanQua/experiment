import torch
import time
import math
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import sys
import utils.tools as tool

# 设置系统设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 获取歌词数据
train_set, vocab_size, idx_to_char, char_to_idx = tool.read_jay_lyrics()

# 初始化模型参数;相关超参数
# 输入数量，隐藏数量，输出数量,隐藏单元的个数是一个超参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
batch_size = 32
num_steps = 35  # 在批次内的序列步长
lr = 1e2  # 学习率
num_epochs = 250  # 训练周期
clipping_theta = 1e-2  # 梯度裁剪阈值
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

# 采样方式,随机采样(random),相邻采样(adjacent)
sample_type = 'random'
if sample_type is 'random':
    sample_data = tool.seq_random_sample
elif sample_type is 'adjacent':
    smaple_data = tool.seq_adjacent_sample

# 初始化参数
def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device,
                          dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))
    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])

# 获取参数;定义损失函数
params = get_params()
loss = nn.CrossEntropyLoss()

# 训练模型并预测
for epoch in range(num_epochs):
    # 若使用相邻采样，在epoch开始时初始化隐藏状态
    if sample_type is 'adjacent':
        state = tool.init_rnn_state(batch_size, num_hiddens, device)
    # 通过采样方式获取采样数据
    data = sample_data(train_set, batch_size, num_steps, device)
    # 定义变量
    l_sum, n, start = 0.0, 0, time.time()

    for X, Y in data:
        print('训练样本形状：', X.size(), Y.size())
        # 如使用随机采样，在每个小批量更新前初始化隐藏状态
        if sample_type is 'random':
            state = tool.init_rnn_state(batch_size, num_hiddens, device)
        else:
            # 否则需要使用detach函数从计算图分离隐藏状态,这是为了使模型参数的梯度只依赖
            # 一次迭代读取的小批量序列（防止梯度计算开销太大）
            for s in state:
                s.detach_()
        # 将采样的数据转化成词向量
        inputs = tool.seq_one_hot(X, vocab_size)
        print('转化后的词向量形状：', inputs[0].size())
        # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵,即一批次的输出
        (outputs, state) = tool.rnn(inputs, state, params)
        # 拼接之后的形状为(num_steps*batch_size, vocab_size)
        outputs = torch.cat(outputs, dim=0)
        # Y的形状是(batch_size, num_steps),装置后再变成长度为batch*num_steps的向量，这样
        # 跟输出的行时一一对应
        y = torch.transpose(Y, 0, 1).contiguous().view(-1)

        # 使用交叉熵计算平均分类误差
        l = loss(outputs, y.long())

        # 梯度清0；求梯度
        if params[0].grad is not None:
            for params in params:
                params.grad.data.zero_()
        l.backward()

        # 梯度裁剪
        tool.grad_clipping(params, clipping_theta, device)
        # 更新参数,因为误差已经去过均值，梯度不用再做平均
        tool.sgd(params, lr, 1)

        # 统计损失值；样本数量
        l_sum += l.item()*y.size(0)
        n += y.size(0)

    # 输出信息
    if (epoch+1) % pred_period == 0:
        print('epoch:{}, perplexity:{}, time:{}'.format(epoch+1, math.exp(l_sum/n),
                                                        time.time() - start))
        for prefix in prefixes:
            print(' -', tool.rnn_predict(prefix, pred_len, tool.rnn, params, tool.init_rnn_state,
                                         num_hiddens, vocab_size, device, idx_to_char, char_to_idx))

