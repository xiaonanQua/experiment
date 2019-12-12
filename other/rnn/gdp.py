from pandas_datareader import wb
from torch import nn
import torch

class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x[:, :, None]
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = x[:, :, 0]
        return x

# 读取数据；并进行归一化
data = wb.download(indicator='NY.GDP.PCAP.KD', country=['US', 'CN'], start=1970, end=2016)
df = data.unstack().T
df.index = df.index.droplevel(0)
num_year, num_sample = df.shape
countries = df.columns
years = df.index
print(df)
# df_scaled = df / df.loc[2000]
# print(df_scaled)
# 确定训练集和测试集
# train_seq_len = sum((years >= 1970)&(years <= 2000))
# test_seq_len = sum(years > 2000)
# print('训练集长度＝{}，测试集长度＝{}'.format(train_seq_len, test_seq_len))

# 确定训练使用的特征和标签
inputs = torch.tensor(df[:-1].values, dtype=torch.float32)
labels = torch.tensor(df[1:], dtype=torch.float32)



