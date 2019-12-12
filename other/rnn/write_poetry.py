import torch
from torch import nn
import numpy as np

embedding = nn.Embedding(10, 2)
words = torch.arange(0, 6).view(3, 2).long()
output = embedding(words)
print(words.size())
print(embedding(words).size())
print(embedding.weight.size())

# 加载数据
file_path = '/home/xiaonan/Dataset/tang/tang.npz'
files = np.load(file_path)
# data = files['data']
ix2word = files['word2ix']
print(files['data'])
# print(data, files.files)