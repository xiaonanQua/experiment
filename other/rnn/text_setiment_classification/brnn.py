import torch.nn as nn
import torch


class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        """
        构建双向循环结构
        :param vocab: 数据词典
        :param embed_size:词向量的大小
        :param num_hiddens: 隐藏层的数量
        :param num_layers: 层数
        """
        super(BiRNN, self).__init__()

        # 嵌入层(将文本数据表示成词向量)
        self.embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size)

        # LSTM实例，序列编码的隐藏层，将bidirectional设置成True即为双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=num_hiddens,
                               num_layers=num_layers, bidirectional=True)

        # Linear实例，生成分类结果的输出层。
        # 初始时间步和最终时间步的隐藏状态作为全连接的输入（向量拼接），对特征进行解码。
        self.decoder = nn.Linear(4*num_hiddens, 2)

    def forward(self, inputs):
        # inputs的输入形状是（批次大小，词数），使用permute将数据的维度位置置换一下。
        # 输出形状是（词数，批次大小，词向量）
        embeddings = self.embedding(inputs.permute(1, 0))

        # 将词向量数据传入LSTM结构中
        # 输出形状（词数，批量大小，2*隐藏单元个数）
        outputs, _ = self.encoder(embeddings)

        # 连接（向量拼接）初始时间步和最终时间步的隐藏状态作为全连接输入。
        # 形状为：[批量大小， 4*隐藏单元个数]
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        out = self.decoder(encoding)

        return out







