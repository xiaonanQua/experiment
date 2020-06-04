from other.rnn.text_setiment_classification.data import ProcessACLData
from other.rnn.text_setiment_classification.brnn import BiRNN
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe
import torch, os

# 设置GPU环境
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class SemanticClassification:
    def __init__(self):
        # 定义相关参数
        self.batch_size = 64  # 批次大小
        self.num_workers = 8  # 线程数量
        self.embedded_size = 100  # 词向量大小
        self.num_hiddens = 100  # 隐藏层数量
        self.num_layers = 2  # 堆叠LSTM的层数
        self.num_classes = 2  # 类别数量
        self.name_classes = ['neg', 'pos']  # 类别名称
        self.pretrained = True  # 是否使用预训练的词向量
        self.lr = 0.01  # 学习率
        self.num_epochs = 20  # 批次数量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.checkpoints = 'text.pth'

        # 定义处理对象的实体类
        data = ProcessACLData()

        # 获得词典
        self.vocab_data = data.get_vocab_imdb()

        # 获取训练、测试数据集及其词典
        self.train_set = data.get_dataset(data.train_dir, self.vocab_data)
        self.test_set= data.get_dataset(data.test_dir, self.vocab_data)
        print('训练数据大小：{}'.format(len(self.train_set)))
        print('测试数据大小：{}'.format(len(self.test_set)))

        # 获得数据加载器
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size,
                                       shuffle=True, num_workers=8)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size,
                                      num_workers=8)

        # 定义训练和测试网络结构
        self.net = BiRNN(vocab=self.vocab_data, embed_size=self.embedded_size,
                               num_hiddens=self.num_hiddens, num_layers=self.num_layers)

        # 定义GloVe对象来加载预训练词向量,维度要与网络中embedding大小一致
        glove = GloVe(name='6B', dim=self.embedded_size, cache='glove')

        # 加载预训练的值
        if self.pretrained:
            self.net.embedding.weight.data.copy_(
                self.load_pretrained_embedding(self.vocab_data.itos, glove)
            )
            # 取消求梯度
            self.net.embedding.weight.requires_grad = False

    def train(self, net):
        """
        训练
        :return:
        """
        # 定义Adam优化器和交叉熵损失函数
        if self.pretrained:
            # 过滤掉预训练的词向量的权重，只训练没有包含的值
            self.optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,
                                                     net.parameters()), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()

        print('training stage....')
        # 将网络结构调成训练模式；将网络放置到GPU上；初始化梯度张量
        net.cuda(device=self.device)
        self.optimizer.zero_grad()

        # 周期遍历
        for epoch in range(self.num_epochs):

            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 20)

            # 调成训练模式
            net.train()
            # 定义准确率变量，损失值，批次数量,样本总数量;最好精确率
            train_acc = 0.0
            train_loss = 0.0
            num_batch = 0
            num_samples = 0
            best_acc = 0

            # 进行每周期的网络的训练
            for index, data in enumerate(self.train_loader, start=0):
                # 获取每批次的训练数据、并将训练数据放入GPU中
                words, labels = data
                words = words.to(self.device)
                labels = labels.to(self.device)

                # 推理输出网络预测值，并使用softmax使预测值满足0-1概率范围；计算损失函数值
                outputs = net(words)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                loss = self.criterion(outputs, labels)

                # 计算每个预测值概率最大的索引（下标）
                preds = torch.argmax(outputs, dim=1)

                # 计算批次的准确率，预测值中预测正确的样本占总样本的比例
                # 统计准确率、损失值、批次数量
                acc = torch.sum(preds == labels).item()
                train_acc += acc
                train_loss += loss
                num_batch += 1
                num_samples += words.size(0)

                # 计算梯度、更新参数、重置梯度张量
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # 输出一定次数的损失和精度情况
                if (index + 1) % 10 == 0:
                    # 输出损失值和精度值
                    print('   batch:{}, batch_loss:{:.4f}, batch_acc:{:.4f}\n'.
                          format(index, loss, acc / words.size(0)))

            # 计算训练的准确率和损失值
            train_acc = train_acc / num_samples
            train_loss = train_loss / num_batch

            # 进行验证
            valid_acc, valid_loss = self.eval(net, self.test_loader,
                                              self.criterion)
            # 输出损失值和精度值
            print('epoch:{} -------\n train loss:{:.4f}, train acc:{:.4f}\n '
                  'valid loss:{:.4f}, valid acc:{:.4f}\n'.
                  format(epoch,train_loss, train_acc, valid_loss, valid_acc))

            # 选出最好的模型参数
            if valid_acc > best_acc:
                # 更新最好精度、保存最好的模型参数
                best_acc = valid_acc
                torch.save(net.state_dict(), self.checkpoints)
                print('epoch:{}, update model...'.format(epoch))
            print()

    def eval(self, net, valid_loader, criterion):
        """
        验证
        :param net: 网络结构
        :param valid_loader: 验证集加载器
        :param criterion: 损失函数
        :return:
        """
        print('  valid stage...')
        # 将网络结构调成验证模式;所有样本的准确率、损失值;统计批次数量;
        net.eval()
        net.cuda(device=self.device)
        valid_acc = 0.0
        valid_loss = 0.0
        num_batch = 0
        num_samples = 0

        # 进行测试集的测试
        with torch.no_grad():  # 不使用梯度，减少内存占用
            for index, dataset in enumerate(valid_loader, start=0):
                data, labels = dataset
                # 将测试数据放入GPU上
                data, labels = data.to(self.device), labels.to(self.device)
                # 推理输出网络预测值，并使用softmax使预测值满足0-1概率范围
                outputs = net(data)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                # 计算每个预测值概率最大的索引（下标）；计算损失值
                pred = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, labels)

                # 统计真实标签和预测标签的对应情况;计算损失
                valid_acc += torch.sum((pred == labels)).item()
                valid_loss += loss
                num_batch += 1
                num_samples += data.size(0)

        # 计算测试精度和损失值
        valid_acc = valid_acc / num_samples
        valid_loss = valid_loss / num_batch

        return valid_acc, valid_loss

    def test(self, net, test_loader):
        print('test stage...\n')
        # 加载模型权重、将网络放入GPU
        if os.path.exists(self.checkpoints):
            net.load_state_dict(torch.load(self.checkpoints))
            print('load model argument...')
        net.to(self.device)

        # 将网络结构调成验证模式、定义准确率、标签列表和预测列表
        net.eval()
        correct = 0
        targets, preds = [], []

        # 进行测试集的测试
        with torch.no_grad():  # 不使用梯度，减少内存占用
            for data, labels in test_loader:
                # 将测试数据放入GPU上
                data, labels = data.to(self.device), labels.to(self.device)
                # 推理输出网络预测值，并使用softmax使预测值满足0-1概率范围
                outputs = net(data)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                pred = torch.argmax(outputs, dim=1)

                # 计算每个预测值概率最大的索引（下标）；统计真实标签和对应预测标签
                correct += torch.sum((pred == labels)).item()
                targets += list(labels.cpu().numpy())
                preds += list(pred.cpu().numpy())

        # 计算测试精度和混淆矩阵
        test_acc = 100. * correct / len(test_loader.dataset)
        # confusion_mat = metrics.confusion_matrix(targets, preds)
        # confusion_mat = confusion_matrix(targets, preds)
        # print('numbers samples:{}, test accuracy:{},\nconfusion matrix:\n{}'.
        #       format(len(test_loader.data2), test_acc, confusion_mat))
        print('numbers samples:{}, test accuracy:{},\n'.
              format(len(test_loader.dataset), test_acc))
        return test_acc

    def predict(self, net, vocab, sentence):
        """
        预测句子的情感
        :param net: 网络结构
        :param vocab: 词典数据
        :param sentence: 预测句子
        :return:
        """
        print('predict stage...\n')
        # 加载模型权重、将网络放入GPU
        if os.path.exists(self.checkpoints):
            net.load_state_dict(torch.load(self.checkpoints))
            print('load model argument...')

        # 将网络结构调成验证模式
        net.eval()

        # 将数据转化成词向量
        vector = torch.tensor([vocab.stoi[word] for word in sentence])
        vector = vector.view(1, -1)

        # 模型预测
        output = net(vector)
        label = torch.argmax(output, dim=1).cpu().tolist()

        # 输出
        print('data2:{}, label:{}'.format(sentence, self.name_classes[label[0]]))

    def load_pretrained_embedding(self, word_vocab, pretrained_vocab):
        """
        从GloVe中预训练好的pretrained_vocab中提取当前vocab对应的词向量
        :param word_vocab: 当前数据集的词典,['ship',...]
        :param pretrained_vocab:GloVe中预训练的词向量
        :return:
        """
        # 初始当前数据的词向量
        embedding = torch.zeros(len(word_vocab), pretrained_vocab.vectors[0].shape[0])
        # 统计不包含的单词
        num_out = 0

        # 遍历当前词典
        for i, word in enumerate(word_vocab):
            # 若单词不在GloVe中，则报出异常
            try:
                # word对应于GloVe中word的索引,并根据索引替换掉向量
                idx = pretrained_vocab.stoi[word]
                embedding[i, :] = pretrained_vocab.vectors[idx]
            except KeyError:
                num_out += 1
            print('\r{}'.format(i), end='  ')

        # 输出不包含的单词数量
        if num_out > 0:
            print('有{}单词不包含'.format(num_out))

        return embedding


if __name__ == '__main__':
    data_1 = ['this', 'movie', 'is', 'so', 'great']
    data_2 = ['this', 'movie', 'is', 'so', 'bad']

    text = SemanticClassification()
    # text.train(text.net)
    text.predict(text.net, text.vocab_data, data_2)







