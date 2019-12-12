from torch import nn
import torch

class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # 将bidirectional设置为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=True)
        # 初始时间步和最终时间步的隐状态作为全连接层输入
        self.decoder = nn.Linear(4*num_hiddens, 2)

    def forward(self, inputs):
        embedding = self.embedding(inputs.permute(1, 0))
        outputs,_ = self.encoder(embedding)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs